path = './dataset/images/'

def pil_loader(path):    
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


ds = torchvision.datasets.DatasetFolder('/content/imgs', 
                                        loader=pil_loader, 
                                        extensions=('.png'), 
                                        transform=t)

print(ds)
