import os
import gc
import numpy as np

# 1. 환경 변수 설정
os.environ["WGPU_BACKEND_TYPE"] = "Vulkan"
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
# 2. 데이터 생성 함수 (3면 격자 추가)
# ---------------------------------------------------------
def plot_wireframe(X, Y, Z):
    rows, cols = X.shape
    Y_min, Y_max = Y.min(), Y.max()
    # 분모가 0이 되는 것을 방지
    denom = Y_max - Y_min
    if denom == 0:
        Y_norm = np.zeros_like(Y)
    else:
        Y_norm = (Y - Y_min) / denom
    
    # R, G, B를 Y_norm과 동일한 크기의 배열로 생성
    # Y_norm이 배열이므로 아래 연산 결과도 배열이 됩니다.
    R = (0.1 * (1 - Y_norm)).astype(np.float32)
    G = (0.6 * Y_norm + 0.2).astype(np.float32)
    B = np.full_like(Y_norm, 1.0).astype(np.float32) # 전체를 1.0으로 채운 배열
    
    # 이제 ravel()이 안전하게 작동합니다.
    vertices = np.stack([
        X.ravel(), Y.ravel(), Z.ravel(), 
        R.ravel(), G.ravel(), B.ravel()
    ], axis=1).astype(np.float32)
    
    idx = np.arange(rows * cols).reshape(rows, cols)
    h_lines = np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], axis=1)
    v_lines = np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], axis=1)
    indices = np.vstack([h_lines, v_lines]).astype(np.uint16).ravel()
    
    return vertices.flatten(), indices

def create_full_grid_data(size=10, divisions=10):
    """XZ(바닥), XY(뒷면), YZ(옆면) 3개 면의 격자 생성"""
    vertices = []
    indices = []
    step = size / divisions
    start = -size / 2
    end = size / 2
    
    curr_idx = 0
    def add_line(p1, p2, color):
        nonlocal curr_idx
        vertices.extend([*p1, *color])
        vertices.extend([*p2, *color])
        indices.extend([curr_idx, curr_idx + 1])
        curr_idx += 2

    # 색상 정의
    c_xz = (0.3, 0.3, 0.3) # 바닥 (회색)
    c_xy = (0.2, 0.3, 0.2) # 뒷면 (연녹조)
    c_yz = (0.3, 0.2, 0.2) # 옆면 (연적조)

    for i in range(divisions + 1):
        d = start + i * step
        # 1. XZ 평면 (바닥: y=start)
        add_line([d, start, start], [d, start, end], c_xz)
        add_line([start, start, d], [end, start, d], c_xz)
        
        # 2. XY 평면 (뒷벽: z=start)
        add_line([d, start, start], [d, end, start], c_xy)
        add_line([start, d, start], [end, d, start], c_xy)
        
        # 3. YZ 평면 (옆벽: x=start)
        add_line([start, d, start], [start, d, end], c_yz)
        add_line([start, start, d], [start, end, d], c_yz)

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint16)

# ---------------------------------------------------------
# 3. WGSL 셰이더 및 앱 클래스
# ---------------------------------------------------------
shader_source = """
struct CameraUniform { view_proj: mat4x4<f32> };
@group(0) @binding(0) var<uniform> camera: CameraUniform;
struct VertexInput { @location(0) position: vec3<f32>, @location(1) color: vec3<f32> };
struct VertexOutput { @builtin(position) clip_pos: vec4<f32>, @location(0) color: vec3<f32> };

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = camera.view_proj * vec4<f32>(model.position, 1.0);
    out.color = model.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"""

class App:
    def __init__(self, canvas, vertex_data, index_data):
        self.canvas = canvas
        self.yaw, self.pitch, self.camera_radius = np.radians(-45), np.radians(20), 15.0
        self.is_dragging = False
        self.last_mouse_pos = None

        canvas.add_event_handler(self.on_pointer_down, "pointer_down")
        canvas.add_event_handler(self.on_pointer_up, "pointer_up")
        canvas.add_event_handler(self.on_pointer_move, "pointer_move")
        canvas.add_event_handler(self.on_wheel, "wheel")
        canvas.add_event_handler(self.on_resize, "resize")

        self.adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.device = self.adapter.request_device_sync()
        self.present_context = canvas.get_context("wgpu")
        self.present_format = self.present_context.get_preferred_format(self.adapter)
        self.present_context.configure(device=self.device, format=self.present_format)

        # 메인 그래프 버퍼
        self.index_count = len(index_data)
        self.vertex_buffer = self.device.create_buffer_with_data(data=vertex_data, usage=wgpu.BufferUsage.VERTEX)
        self.index_buffer = self.device.create_buffer_with_data(data=index_data, usage=wgpu.BufferUsage.INDEX)

        # 3면 격자 버퍼 생성
        grid_verts, grid_indices = create_full_grid_data(size=10, divisions=10)
        self.grid_index_count = len(grid_indices)
        self.grid_vertex_buffer = self.device.create_buffer_with_data(data=grid_verts, usage=wgpu.BufferUsage.VERTEX)
        self.grid_index_buffer = self.device.create_buffer_with_data(data=grid_indices, usage=wgpu.BufferUsage.INDEX)

        self.camera_buffer = self.device.create_buffer(size=64, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        shader = self.device.create_shader_module(code=shader_source)

        self.pipeline = self.device.create_render_pipeline(
            layout="auto",
            vertex={
                "module": shader, "entry_point": "vs_main",
                "buffers": [{"array_stride": 24, "attributes": [
                    {"format": wgpu.VertexFormat.float32x3, "offset": 0, "shader_location": 0},
                    {"format": wgpu.VertexFormat.float32x3, "offset": 12, "shader_location": 1},
                ]}]
            },
            primitive={"topology": wgpu.PrimitiveTopology.line_list},
            depth_stencil={"format": wgpu.TextureFormat.depth32float, "depth_write_enabled": True, "depth_compare": wgpu.CompareFunction.less},
            fragment={"module": shader, "entry_point": "fs_main", "targets": [{"format": self.present_format}]},
        )
        self.camera_bind_group = self.device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": {"buffer": self.camera_buffer, "offset": 0, "size": 64}}]
        )
        self.depth_texture = None

    def on_pointer_down(self, e): 
        if e["button"] == 1: self.is_dragging, self.last_mouse_pos = True, (e["x"], e["y"])
    def on_pointer_up(self, e): 
        if e["button"] == 1: self.is_dragging = False
    def on_pointer_move(self, e):
        if self.is_dragging:
            dx, dy = e["x"] - self.last_mouse_pos[0], e["y"] - self.last_mouse_pos[1]
            self.yaw += dx * 0.005
            self.pitch = np.clip(self.pitch + dy * 0.005, -1.5, 1.5)
            self.last_mouse_pos = (e["x"], e["y"])
            self.canvas.request_draw()
    def on_wheel(self, e):
        self.camera_radius = np.clip(self.camera_radius + e["dy"] * 0.02, 2.0, 50.0)
        self.canvas.request_draw()
    def on_resize(self, e): self.canvas.request_draw()

    def draw_frame(self):
        current_texture = self.present_context.get_current_texture()
        w, h, _ = current_texture.size
        if w == 0 or h == 0: return

        proj = perspective_rh(np.radians(45), w/h, 0.1, 100.0)
        eye = np.array([
            self.camera_radius * np.cos(self.pitch) * np.cos(self.yaw),
            self.camera_radius * np.sin(self.pitch),
            self.camera_radius * np.cos(self.pitch) * np.sin(self.yaw)
        ], dtype=np.float32)
        view = look_at_rh(eye, np.array([0,0,0], dtype=np.float32), np.array([0,1,0], dtype=np.float32))
        self.device.queue.write_buffer(self.camera_buffer, 0, (proj @ view).T.astype(np.float32).tobytes())

        if self.depth_texture is None or self.depth_texture.size[:2] != (w, h):
            self.depth_texture = self.device.create_texture(size=(w, h, 1), usage=wgpu.TextureUsage.RENDER_ATTACHMENT, format=wgpu.TextureFormat.depth32float)
            self.depth_view = self.depth_texture.create_view()

        encoder = self.device.create_command_encoder()
        render_pass = encoder.begin_render_pass(
            color_attachments=[{"view": current_texture.create_view(), "clear_value": (0.02, 0.02, 0.05, 1), "load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.store}],
            depth_stencil_attachment={"view": self.depth_view, "depth_clear_value": 1.0, "depth_load_op": wgpu.LoadOp.clear, "depth_store_op": wgpu.StoreOp.store}
        )
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.camera_bind_group, [])

        # 1. 3면 격자 그리기
        render_pass.set_vertex_buffer(0, self.grid_vertex_buffer, 0, self.grid_vertex_buffer.size)
        render_pass.set_index_buffer(self.grid_index_buffer, wgpu.IndexFormat.uint16, 0, self.grid_index_buffer.size)
        render_pass.draw_indexed(self.grid_index_count, 1, 0, 0, 0)

        # 2. 메인 그래프 그리기
        render_pass.set_vertex_buffer(0, self.vertex_buffer, 0, self.vertex_buffer.size)
        render_pass.set_index_buffer(self.index_buffer, wgpu.IndexFormat.uint16, 0, self.index_buffer.size)
        render_pass.draw_indexed(self.index_count, 1, 0, 0, 0)

        render_pass.end()
        self.device.queue.submit([encoder.finish()])

    def cleanup(self):
        self.device = None
        gc.collect()

def main():
    x = np.linspace(-4, 4, 50)
    z = np.linspace(-4, 4, 50)
    X, Z = np.meshgrid(x, z)
    # 물결 모양 함수
    Y = np.sin(np.sqrt(X**2 + Z**2)) 
    
    vertex_data, index_data = plot_wireframe(X, Y, Z)
    canvas = RenderCanvas(title="WGPU 3D Plot - 3 Sided Grid", size=(1000, 800))
    app = App(canvas, vertex_data, index_data)
    canvas.request_draw(app.draw_frame)
    try:
        loop.run()
    except KeyboardInterrupt: pass
    finally:
        app.cleanup()
        os._exit(0)

if __name__ == "__main__":
    main()