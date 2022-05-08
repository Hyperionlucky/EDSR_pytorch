import os


base_dir = '/home/ldx/DEM/ESDR_Pytorch/EDSR_pytorch/Dataset_USA'

scale = [2,3]
mode = 'train'

for scale in scale:
    data_path = os.path.join(os.path.abspath(base_dir), mode + '_' + str(scale) + 'x_slope.txt')
    with open(data_path,"r", encoding='utf-8') as f1, open(data_path.replace("slope","flow"),"w") as f2:
        for line in f1:
            samples = ""
            # samples = samples + line.strip().split(" ")[0:2]
            flow_path = line.strip().split(" ")[2].replace("Slope","Flow")
            flow_path = flow_path.replace(str(192//scale)+"_"+str(192//scale)+"_","192_192_")
            samples = line.strip().replace("cgd","ldx").split(" ")[0] + " " + line.strip().replace("cgd","ldx").split(" ")[1]+ " " + flow_path.replace("cgd","ldx")
            f2.write(samples+"\n")
        