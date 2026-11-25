import sys
sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import re
import time
DEVICE = 'cuda'



def Mode(choice,vis,pcd):
   if choice =="tensor" :
    print("cool")
    center = np.array([-5.1257901191711426,
		-6.170992374420166,
		19.951877593994141], dtype=np.float32)
    eye = np.array([-3.8584733009338379,
            -1.4617710113525391,
            10.130702972412109], dtype=np.float32)
    up = np.array([-0.27111944556236267,
            -0.85382193326950073,
            -0.44438990950584412], dtype=np.float32)
    vis.scene.camera.look_at(center, eye, up)







   else :
    vis.create_window(width=1920,height=1080)
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(500, 3)) # initial values 
    vis.add_geometry(pcd)  # adds the point to be changed later 
    grid=np.random.rand(1920,1080)  
    map_3D(10,10,250,20,250,grid,vis,pcd)  
    vis.reset_view_point(True)
    ctr = vis.get_view_control()
    ctr.set_front([ 0.32409266602784592, 0.54884662226404624, -0.77053963496780009])
    ctr.set_lookat([-14.533711109183692, -12.221605356947153, 27.783942222595215])
    ctr.set_up([0.028348084328648798, -0.81977062422281888, -0.57198995600991631 ])
    ctr.set_zoom(0.4)
 



def load_Calibration(file_text):
    with open(file_text, "r", encoding="utf-8") as file:
        text=file.readline()
        Cam0 = [float(num) for num in re.findall(r'-?\d+(?:\.\d+)?', text)]
        text=file.readline()
        Cam1 = [float(num) for num in re.findall(r'-?\d+(?:\.\d+)?', text)]
        text=file.readline()
        text=file.readline()
        baseline= [float(num) for num in re.findall(r'-?\d+(?:\.\d+)?', text)]

    return Cam0[1],Cam0[3],Cam0[6],Cam1[3],Cam1[6],baseline[0]
        


#Creates a point cloud of of the photo black and white
def map_3D(focal_length, base_line, C0x,C1x, Cy, flow_up,vis,pcd):
    
    if isinstance(flow_up, torch.Tensor):
      flow_up = flow_up.detach().cpu().numpy()

    rows,col=flow_up.shape
    coordinates=np.zeros((3,rows,col))
    disparity=(-flow_up-C0x+C1x)
    disparity=np.where(abs(disparity)<0.001,0.001,disparity)
    coordinates[2]=focal_length*base_line/abs(disparity)
    yindex,xindex=np.indices(flow_up.shape)
    coordinates[0]=(xindex-C0x)*coordinates[2]/focal_length
    coordinates[1]=(yindex-Cy)*coordinates[2]/focal_length

    pointcloud=coordinates.transpose(1, 2, 0).reshape(-1, 3).astype(np.float64)

    pcd.points = o3d.utility.Vector3dVector(pointcloud)

    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()




#Creates a point cloud of of the photo with colour 
# first 5 inputs are dtermined by calibration 
def map_3D_colour(focal_length, base_line, C0x,C1x, Cy, flow_up,vis,pcd,image):
    
  if isinstance(flow_up, torch.Tensor):
    flow_up = flow_up.detach().cpu().numpy()


    disparity=(-flow_up-C0x+C1x)  
    disparity=np.where(abs(disparity)<0.001,0.001,disparity) # clamping the value 
  
    Z=focal_length*base_line/abs(disparity)  # depth 
  
    yindex,xindex=np.indices(flow_up.shape)
    X=(xindex-C0x)*Z/focal_length # X distance 
    Y=(yindex-Cy)*Z/focal_length # Y distance 


   
    #Color field 
    pointcloud=np.column_stack((X.ravel(), Y.ravel(), Z.ravel())).astype(np.float64)
    colours=image.reshape(-1,3).astype(np.float64)/255

    # filtering of point cloud 
    # temp_pcd   = o3d.geometry.PointCloud()
    # temp_pcd.points = o3d.utility.Vector3dVector(pointcloud)
    # temp_pcd.colors = o3d.utility.Vector3dVector(colours) # add coloured points 
    # clean_pcd,_= temp_pcd.remove_radius_outlier(nb_points=8, radius=0.05)
    # clean_pcd,_= clean_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)


    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    pcd.colors = o3d.utility.Vector3dVector(colours) # add coloured points 
    # pcd.points = clean_pcd.points
    # pcd.colors = clean_pcd.colors

    #updates the point cloud 
    vis.update_geometry(pcd) 
    vis.poll_events()
    vis.update_renderer()





def map_3D_colour_tensor(focal_length, base_line, C0x,C1x, Cy, flow_up,vis,pcd,image):
    
    device = flow_up.device
    rows, cols = flow_up.shape


    disparity = torch.clamp(torch.abs(-flow_up - C0x + C1x), min=0.001)
    Z = (focal_length * base_line) / abs(disparity)



    y_idx, x_idx = torch.meshgrid(
    torch.arange(rows, device=device, dtype=flow_up.dtype),
    torch.arange(cols, device=device, dtype=flow_up.dtype),
    indexing="ij",)

    X = (x_idx - C0x) * Z / focal_length
    Y = (y_idx - Cy) * Z / focal_length
    
    #Color field 
    pointcloud = torch.stack((X, Y, Z), dim=-1).reshape(-1,3).contiguous()

    img_torch = image.to(device=device, dtype=torch.float32)
    colors_torch = (img_torch.reshape(-1,3)/ 255.0).contiguous()

    pcd.point.positions = o3d.core.Tensor.from_dlpack(pointcloud)
    pcd.point.colors = o3d.core.Tensor.from_dlpack(colors_torch)



    cpu_pcd = o3d.t.geometry.PointCloud(o3d.core.Device("CPU:0"))
    cpu_pcd.point.positions = o3d.core.Tensor(pointcloud.cpu().numpy())
    cpu_pcd.point.colors = o3d.core.Tensor(colors_torch.cpu().numpy())

    
    
    # clean_pcd, _ = cpu_pcd.remove_statistical_outliers(
    #     nb_neighbors=10, std_ratio=2.0
    # )
    # clean_pcd, _ = cpu_pcd.remove_radius_outliers(
    #     nb_points=8, search_radius=0.05
    # )
 
    #updates the point cloud 
    vis.remove_geometry("pcd")
    vis.add_geometry("pcd", cpu_pcd)
    if not gui.Application.instance.run_one_tick():
     return False
     
    return True




def RGB_image_tensor(image):
    
 RGB_matrix = image[0].permute(1, 2, 0)

 if RGB_matrix.dtype != np.uint8:
    RGB_matrix = (RGB_matrix).clip(0,255).to(torch.float32)

 return  RGB_matrix.contiguous()







#handles getting the rgb values of the image 
def RGB_image(image):
    
 RGB_matrix = image[0].permute(1, 2, 0).cpu().numpy()
 if RGB_matrix.dtype != np.uint8:
    RGB_matrix = (RGB_matrix).clip(0,255).astype(np.float64)

 RGB_matrix = np.ascontiguousarray(RGB_matrix)
 return  RGB_matrix
   


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args,vis):
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        text =  sorted(glob.glob(args.text, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")
        for (imfile1, imfile2,values) in tqdm(list(zip(left_images, right_images,text))):

            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            if args.mode == "tensor":
             rgb=RGB_image_tensor(image1)
            else :
             rgb=RGB_image(image1)

            focal_length,C0x,Cy,C1x,_,baseline=load_Calibration(values)
            baseline=baseline/1000

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
            flow_up = padder.unpad(flow_up).squeeze()
                  
            #if(len(text)):
            #  #map_3D(1,1,1,1,1,flow_up,vis,pcd)
            if args.mode == "tensor":
             map_3D_colour_tensor(focal_length,baseline,C0x,C1x,Cy,flow_up,vis,pcd,rgb)
            else:
             map_3D_colour(focal_length,baseline,C0x,C1x,Cy,flow_up,vis,pcd,rgb)
               
            # file_stem = Path(imfile1).parent.name   # one folder per scene (Middlebury layout)
            # if args.save_numpy:
            #     np.save(output_directory / f"{file_stem}.npy", flow_up.cpu().numpy().squeeze())
            # plt.imsave(output_directory / f"{file_stem}.png", -flow_up.cpu().numpy().squeeze(), cmap='jet')
  



##Update them in batches an than show them on the open3d model so that 


def Video_render(args,vis):
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        text =  sorted(glob.glob(args.text, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")
        for (imfile1, imfile2,values) in tqdm(list(zip(left_images, right_images,text))):
   
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            rgb=RGB_image(image1)

            focal_length,C0x,Cy,C1x,_,baseline=load_Calibration(values)
            baseline=baseline/1000

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
            flow_up = padder.unpad(flow_up).squeeze()
                  
            if(len(text)):
            #  map_3D(1,1,1,1,1,flow_up,vis,pcd)
             map_3D_colour(focal_length,baseline,C0x,C1x,Cy,flow_up,vis,pcd,rgb)
            
            # file_stem = Path(imfile1).parent.name   # one folder per scene (Middlebury layout)
            # if args.save_numpy:
            #     np.save(output_directory / f"{file_stem}.npy", flow_up.cpu().numpy().squeeze())
            # plt.imsave(output_directory / f"{file_stem}.png", -flow_up.cpu().numpy().squeeze(), cmap='jet')






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('-t', '--text', help="calibration_text", default="none")
    parser.add_argument('-m', '--mode', help="mode for calculation", default="colour")
    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "reg_gather"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args = parser.parse_args()

if args.mode == "tensor":
  app = gui.Application.instance
  app.initialize()
  vis = o3d.visualization.O3DVisualizer("Tensor PointCloud Viewer", 1024, 768)
  pcd = o3d.t.geometry.PointCloud()
  app.add_window(vis)
else:  
  vis = o3d.visualization.Visualizer()
  pcd = o3d.geometry.PointCloud()



Mode(args.mode,vis,pcd)
demo(args,vis)
# Keeps the image open for longer 
if args.mode == "tensor" :
 app.run()
else:
 vis.run()
   
   


