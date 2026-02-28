import os
import gc
import numpy as np

# 1. 환경 변수 설정 (반드시 wgpu 임포트 전에 실행)
# Vulkan 백엔드 강제 사용
os.environ["WGPU_BACKEND_TYPE"] = "Vulkan"
# 하드웨어 관련 Warning 로그를 숨기고 Error만 표시
os.environ["WGPU_LOG_LEVEL"] = "error"

import wgpu
from rendercanvas.auto import RenderCanvas, loop

# ---------------------------------------------------------
# 1. 3D 수학 헬퍼 함수
# ---------------------------------------------------------
def perspective_rh(fovy_rad, aspect, near, far):
    f = 1.0 / np.tan(fovy_rad / 2.0)
    return np.array([
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far / (near - far), (far * near) / (near - far)],
        [0.0, 0.0, -1.0, 0.0]
    ], dtype=np.float32)

def look_at_rh(eye, center, up):
    f = center - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)

    return np.array([
        [s[0], s[1], s[2], -np.dot(s, eye)],
        [u[0], u[1], u[2], -np.dot(u, eye)],
        [-f[0], -f[1], -f[2], np.dot(f, eye)],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)

# ---------------------------------------------------------
# 2. Matplotlib 스타일의 3D 그래프 데이터 생성 함수
# ---------------------------------------------------------
def plot_wireframe(X, Y, Z):
    """
    X, Y, Z 메쉬그리드 배열을 WGPU용 정점/인덱스 버퍼 배열로 변환합니다.
    """
    rows, cols = X.shape
    
    # 높이(Y)를 0~1 사이로 정규화하여 색상(밝기) 매핑에 사용
    Y_min, Y_max = Y.min(), Y.max()
    Y_norm = (Y - Y_min) / (Y_max - Y_min + 1e-6) if Y_max > Y_min else np.zeros_like(Y)
    
    # 현재는 높이에 따라 흑백 그라데이션 적용
    R = Y_norm
    G = Y_norm
    B = Y_norm
    
    # 정점 배열 생성: (X, Y, Z, R, G, B)
    vertices = np.stack([
        X.ravel(), Y.ravel(), Z.ravel(),
        R.ravel(), G.ravel(), B.ravel()
    ], axis=1).astype(np.float32)
    
    # 인덱스 배열 생성 (벡터 연산으로 고속화)
    idx = np.arange(rows * cols).reshape(rows, cols)
    
    # 가로선(가로 방향 점 연결)과 세로선(세로 방향 점 연결)
    h_lines = np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], axis=1)
    v_lines = np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], axis=1)
    
    # Line List 포맷에 맞게 평탄화(flatten)
    indices = np.vstack([h_lines, v_lines]).astype(np.uint16).ravel()
    
    return vertices.flatten(), indices

# ---------------------------------------------------------
# 3. WGSL 셰이더 소스코드
# ---------------------------------------------------------
shader_source = """
struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    out.color = model.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"""

# ---------------------------------------------------------
# 4. 앱 및 상태 관리 클래스
# ---------------------------------------------------------
class App:
    def __init__(self, canvas, vertex_data, index_data):
        self.canvas = canvas
        self.yaw = np.radians(-90.0)
        self.pitch = np.radians(20.0)
        self.camera_radius = 4.0
        
        self.is_dragging = False
        self.last_mouse_pos = None

        # 이벤트 핸들러 등록
        canvas.add_event_handler(self.on_pointer_down, "pointer_down")
        canvas.add_event_handler(self.on_pointer_up, "pointer_up")
        canvas.add_event_handler(self.on_pointer_move, "pointer_move")
        canvas.add_event_handler(self.on_wheel, "wheel")
        canvas.add_event_handler(self.on_resize, "resize")

        # WGPU 초기화
        self.adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.device = self.adapter.request_device_sync()
        
        self.present_context = canvas.get_context("wgpu")
        self.present_format = self.present_context.get_preferred_format(self.adapter)
        self.present_context.configure(device=self.device, format=self.present_format)

        self.depth_texture = None
        self.depth_texture_view = None

        self.index_count = len(index_data)

        # 버퍼 생성 및 데이터 전달
        self.vertex_buffer = self.device.create_buffer_with_data(
            data=vertex_data, usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST
        )
        
        self.index_buffer = self.device.create_buffer_with_data(
            data=index_data, usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST
        )

        self.camera_buffer = self.device.create_buffer(
            size=64, 
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )

        shader = self.device.create_shader_module(code=shader_source)

        self.pipeline = self.device.create_render_pipeline(
            layout="auto", 
            vertex={
                "module": shader,
                "entry_point": "vs_main",
                "buffers": [
                    {
                        "array_stride": 24,
                        "step_mode": wgpu.VertexStepMode.vertex,
                        "attributes": [
                            {"format": wgpu.VertexFormat.float32x3, "offset": 0, "shader_location": 0},
                            {"format": wgpu.VertexFormat.float32x3, "offset": 12, "shader_location": 1},
                        ],
                    }
                ],
            },
            primitive={"topology": wgpu.PrimitiveTopology.line_list},
            depth_stencil={
                "format": wgpu.TextureFormat.depth32float,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [{"format": self.present_format, "blend": {"color": {}, "alpha": {}}}],
            },
        )

        self.camera_bind_group = self.device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": {"buffer": self.camera_buffer, "offset": 0, "size": 64}}],
        )

    def cleanup(self):
        """리소스 명시적 해제 메서드"""
        print("Cleaning up WGPU resources...")
        self.device = None
        self.adapter = None
        gc.collect()

    def on_pointer_down(self, event):
        if event["button"] == 1: 
            self.is_dragging = True
            self.last_mouse_pos = (event["x"], event["y"])

    def on_pointer_up(self, event):
        if event["button"] == 1:
            self.is_dragging = False
            self.last_mouse_pos = None

    def on_pointer_move(self, event):
        if self.is_dragging and self.last_mouse_pos:
            dx = event["x"] - self.last_mouse_pos[0]
            dy = event["y"] - self.last_mouse_pos[1]
            sensitivity = 0.005
            
            self.yaw += dx * sensitivity
            self.pitch += dy * sensitivity
            self.pitch = np.clip(self.pitch, np.radians(-89.0), np.radians(89.0))
            
            self.last_mouse_pos = (event["x"], event["y"])
            self.canvas.request_draw()

    def on_wheel(self, event):
        dy = event["dy"]
        scroll_amount = dy * 0.05
        self.camera_radius = np.clip(self.camera_radius + scroll_amount, 1.0, 20.0)
        self.canvas.request_draw()

    def on_resize(self, event):
        self.canvas.request_draw()

    def draw_frame(self):
        current_texture = self.present_context.get_current_texture()
        w, h, _ = current_texture.size
        
        if w == 0 or h == 0:
            return

        aspect = w / h
        proj = perspective_rh(np.radians(45.0), aspect, 0.1, 100.0)
        
        x = self.camera_radius * np.cos(self.pitch) * np.cos(self.yaw)
        y = self.camera_radius * np.sin(self.pitch)
        z = self.camera_radius * np.cos(self.pitch) * np.sin(self.yaw)
        
        eye = np.array([x, y, z], dtype=np.float32)
        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        view = look_at_rh(eye, center, up)
        view_proj = proj @ view
        view_proj_bytes = view_proj.T.astype(np.float32).tobytes()
        self.device.queue.write_buffer(self.camera_buffer, 0, view_proj_bytes)

        if self.depth_texture is None or self.depth_texture.size[:2] != (w, h):
            self.depth_texture = self.device.create_texture(
                size=(w, h, 1),
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
                dimension=wgpu.TextureDimension.d2,
                format=wgpu.TextureFormat.depth32float,
            )
            self.depth_texture_view = self.depth_texture.create_view()

        color_view = current_texture.create_view()
        encoder = self.device.create_command_encoder()
        
        render_pass = encoder.begin_render_pass(
            color_attachments=[{
                "view": color_view,
                "clear_value": (0.1, 0.2, 0.3, 1.0),
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
            }],
            depth_stencil_attachment={
                "view": self.depth_texture_view,
                "depth_clear_value": 1.0,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
            }
        )

        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.camera_bind_group, [])
        render_pass.set_vertex_buffer(0, self.vertex_buffer, 0, self.vertex_buffer.size)
        render_pass.set_index_buffer(self.index_buffer, wgpu.IndexFormat.uint16, 0, self.index_buffer.size)
        render_pass.draw_indexed(self.index_count, 1, 0, 0, 0)
        render_pass.end()

        self.device.queue.submit([encoder.finish()])

# ---------------------------------------------------------
# 5. 실행 및 종료 처리
# ---------------------------------------------------------
def main():
    # 1. Matplotlib 스타일로 X, Z 영역 및 해상도 정의
    # 해상도를 더 높이고 싶다면 50 대신 더 큰 숫자(예: 100, 200)를 입력하세요.
    x = np.linspace(-5, 5, 50)
    z = np.linspace(-5, 5, 50)
    X, Z = np.meshgrid(x, z)
    
    # 2. 원하는 수학 함수(Y) 정의 (물결 모양 파동)
    R = np.sqrt(X**2 + Z**2)
    Y = np.sin(R)
    
    # 3. 데이터 변환 (Numpy 2D -> WGPU 1D Array)
    vertex_data, index_data = plot_wireframe(X, Y, Z)
    
    # 4. 앱 실행
    canvas = RenderCanvas(title="WGPU-PY 3D Graph (Matplotlib Style)", size=(800, 600))
    app = App(canvas, vertex_data, index_data)
    
    canvas.request_draw(app.draw_frame)
    
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.cleanup()
        os._exit(0)

if __name__ == "__main__":
    main()