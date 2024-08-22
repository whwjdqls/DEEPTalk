import torch
import os
os.environ['PYOPENGL_PLATFORM'] =  'osmesa'
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    HardPhongShader,
    TexturesUV,
    TexturesVertex
)
import nvdiffrast 
import nvdiffrast.torch as dr
from models.flame_models import flame
import numpy as np
import json
from torchvision import transforms
import time
import pyrender

import trimesh
from PIL import Image
import imageio.v2 as imageio   
import cv2
import glob
import subprocess
def get_texture_from_template(template_path, device):
    """get texture from template

    Args:
        template_path (str): path to template
        device (torch.device): device
    return:
        tex (TexturesUV): TexturesUV from template obj file
        Note that there is only one tex. use tex.extend(BS) to get a list of tex
    """
    # get textures from head_template.obj
    verts, faces, aux = load_obj(template_path)
    verts_uvs = aux.verts_uvs.to(device)    # ( V, 2)
    faces_uvs = faces.textures_idx.to(device) # (F, 3)
    tex_maps = aux.texture_images
    image = list(tex_maps.values())[0].to(device)[None]
    tex = TexturesUV(
            verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image
        )
    return tex

#(TODO) add function description
def load_template_mesh(template_path, device) :
    """_summary_

    Args:
        template_path (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """
    verts, faces, aux = load_obj(template_path)
    uv = aux.verts_uvs.to(torch.float32).to(device) #(5118,2)
    uv_idx = faces.textures_idx.to(torch.int32).to(device) #(9976,3)
    mesh_f = faces.verts_idx.to(torch.int32).to(device) #(9976,3)
    tex = aux.texture_images['FaceTexture'].to(device) #(512,512,3)

    return uv, uv_idx, mesh_f, tex

#(TODO) add function description
def render_flame_nvdiff(glctx, vertices, uv, uv_idx, faces, texture, res, device) :
    bst = vertices.shape[0]
    rgbs = []
    # convert order of vertices for uv
    uv = -uv
    for i in range(bst) :
        mesh_v_ = vertices[i].unsqueeze(0)
        rast_out, rast_out_db = dr.rasterize(glctx, mesh_v_, faces, [res, res])
        texc, texd = dr.interpolate(uv[None, ...].contiguous(), rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
        color = dr.texture(texture[None, ...].contiguous(), texc, texd, filter_mode='linear-mipmap-linear')
        color = color * torch.clamp(rast_out[..., -1:], 0, 1)
        rgbs.append(color)
    images = torch.cat(rgbs, dim=0).view(bst,res,res,3)

    return images





def render_flame(config, vertices, faces, textures, device):
    """render flame model
    (note) this is a function that makes BS*T meshes...is it too memory consuming?
    Args:
        config (dict): config file
        vertices (torch.tensor): verticies (BS*T, 5023, 3)
        faces (torch.tensor): template FLAME Faces (BS*T,9976,3)
        textures (TexturesUV): TexturesUV from template obj file length BS*T
        device (torch.device): device
    return:
        images (torch.tensor): rendered images (BS*T, 256, 256, 4)
    """
    BS = vertices.shape[0]
    # get mesh from vertices
    meshes = Meshes(verts=vertices, faces=faces, textures=textures)
    # get camera
    T = torch.tensor([[0., 0.01, 0.2]]).repeat(BS,1)
    R = torch.tensor([[[-1.,  0.,  0.],
                    [ 0.,  1.,  0.],
                    [ 0.,  0., -1.]]]).repeat(BS,1,1)
    znear = torch.tensor(0.01).to(device)
    cameras = FoVPerspectiveCameras(znear=znear, device=device, R=R, T=T)
    # get lights
    lights = PointLights(device=device, location=[[0., 1.,2.]])
    # get rasterization settings
    raster_settings = RasterizationSettings(
        image_size=224, 
        blur_radius=0.0, 
        faces_per_pixel=1,
        bin_size=0
    )

    # get renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras, 
            lights=lights
        )
    )
    # render
    images = renderer(meshes)
    return images

def render_flame_lip(config, vertices, faces, textures, device):
    """render flame model
    (note) this is a function that makes BS*T meshes...is it too memory consuming?
    Args:
        config (dict): config file
        vertices (torch.tensor): verticies (BS*T, 5023, 3)
        faces (torch.tensor): template FLAME Faces (BS*T,9976,3)
        textures (TexturesUV): TexturesUV from template obj file length BS*T
        device (torch.device): device
    return:
        images (torch.tensor): rendered images (BS*T, 88, 88, 4)
    """
    BS = vertices.shape[0]
    # get mesh from vertices
    meshes = Meshes(verts=vertices, faces=faces, textures=textures)
    # get camera
    T = torch.tensor([[0., 0.0475, 0.09]]).repeat(BS,1)
    R = torch.tensor([[[-1.,  0.,  0.],
                    [ 0.,  1.,  0.],
                    [ 0.,  0., -1.]]]).repeat(BS,1,1)
    znear = torch.tensor(0.01).to(device)
    cameras = FoVPerspectiveCameras(znear=znear, device=device, R=R, T=T)
    # get lights
    lights = PointLights(device=device, location=[[0., 1.,2.]])
    # get rasterization settings
    raster_settings = RasterizationSettings(
        image_size=88,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )

    # get renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )
    # render
    images = renderer(meshes)
    return images

def to_lip_reading_image(images):
    """convert to images that are input to lip reading model
    (NOTE) this function is for lip reading PER FRAME
    Args:
        images (torch.tensor): rendered images (BS*T,3, 256, 256) or (BS*T, 3, 88, 88)
    return:
        images (torch.tensor): rendered images (BS*T, 1, 1, 88, 88)
    """

    if images.shape[-1] == 224:
        # images = transforms.functional.crop(images, 140, 85, 88, 88) # (BS*T, 3, 88, 88)
        images = transforms.functional.crop(images, 122, 74, 88, 88)
    grayscaled = transforms.functional.rgb_to_grayscale(images) # (BS*T, 1, 88, 88)
    return grayscaled.unsqueeze(1) # (BS*T, 1, 1, 88, 88)

# (12-14) pyrender not working for NOW..
def pyrender_flame(vertices, faces, image_save_dir):
    """render flame model with pyrender
    (NOTE) faces can be obtained from FLAME.faces
    Args:
        vertices (torch.tensor): verticies (T, 5023, 3)
        faces (torch.tensor): template FLAME Faces (9976,3)
    
    """
    for i in range(vertices.shape[0]):
        vertices_ = vertices[i].detach().cpu().numpy()
        vertex_colors = np.ones([vertices_.shape[0], 4]) * [0.5, 0.5, 0.5, 1.0]
        tri_mesh = trimesh.Trimesh(vertices_, faces, vertex_colors=vertex_colors,
                               face_colors=[0.7, 0.7, 0.7, 1.0], process=False)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=False)
        # create camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0, aspectRatio=1.0)
        camera_pose = np.array([[1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.35],  # Adjust Z position
                                [0.0, 0.0, 0.0, 1.0]])
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene = pyrender.Scene(nodes=[camera_node],
                            bg_color = [0.0, 0.0, 0.0, 1.0])
        scene.add(mesh)

        # create light
        light = pyrender.DirectionalLight(color=[0.8, 0.8, 0.8], intensity=0.5)
        scene.add(light)
        # create renderer
        renderer = pyrender.OffscreenRenderer(viewport_width=1400, viewport_height=1080)
        # render
        color, _ = renderer.render(scene)
        image = Image.fromarray(color)
        # save
        imageio.imwrite(f'{image_save_dir}/{i+1:03d}.png', color)
    return

def make_video_from_images(video_file_path, image_save_dir, fps):
    """make video from a directory of images names 001.png, 002.png, ...
    (NOTE) import imageio.v2 as imageio 
    Args:
        video_file_path (str): path of the video file to be saved
        image_save_dir (str): path to the directory of the images
        fps (int): fps of the video
    """
    png_files = sorted(glob.glob(os.path.join(image_save_dir, '*.png')),key=lambda x: int(os.path.basename(x).split('.')[0]))

    # Create a video writer
    writer = imageio.get_writer(video_file_path, fps=fps, codec='libx264', macro_block_size=None,
                                ffmpeg_params=['-pix_fmt', 'yuv420p', '-crf', '18'])

    for file in png_files:
        # Read the image
        im = imageio.imread(file)        
        writer.append_data(im)
    # Close the video writer
    writer.close()
    print(f"Video created: {video_file_path}")
    
def add_audio_to_video(video_file_path, save_folder, audio_file_path):
    """add audio to video
    (NOTE) ffmpeg must be installed
    Args:
        video_file_path (str): path to the video file
        audio_file_path (_type_): path to the audio file
    """
    wav_file_path = audio_file_path
    video_with_audio_file_path = os.path.basename(video_file_path).split('.')[0] + "_audio.mp4"
    video_with_audio_file_path = f'{save_folder}/{video_with_audio_file_path}'
    
    cmd_command = f"ffmpeg -i {video_file_path} -i {wav_file_path} -c:v copy -c:a aac {video_with_audio_file_path}"
    try:
        result = subprocess.run(cmd_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode("utf-8")
        error = result.stderr.decode("utf-8")

        print("CMD Output:")
        print(output)
        
        if error:
            print("CMD Error:")
            print(error)
    except subprocess.CalledProcessError as e:
        print("Error occurred:", e)