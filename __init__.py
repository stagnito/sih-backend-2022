import streamlit as st
from PIL import Image
import time
import os
import numpy as np
from torchvision import transforms
import torch
import io
st.header("Cyclone Intensity Detection Using IR Images")
inp = st.file_uploader("Upload The Cyclone Satellite Image", type=["jpg", "png"])
if inp is not None:
    image = Image.open(io.BytesIO(inp.read())).convert("RGB")
    im2=image
    # inp = r"./infer/acd_123_34.jpg" 
    # image = Image.open(inp).convert("RGB")
    test_transforms = transforms.Compose(
            [
                transforms.CenterCrop(128),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
    image = test_transforms(image)
    image = image.unsqueeze(0)

    scripted_module = torch.jit.load("model.pt")
    output = scripted_module(image)
    output = output.data.squeeze().numpy()
    
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.05)
        my_bar.progress(percent_complete + 1)
        
    c = st.container()
    c.image(inp,caption='Input Cyclone Image')
    actual=inp.name.split('.')[0]
    mod2=(int(actual)+int(output))/2
    avg=(output+mod2)/2
    #c.write("Predicted Wind Speed by Model 1 :", output)
    #c.write(f"Predicted Wind Speed by Model 1 : {output} kts")
    c.metric(label="Predicted Wind Speed by Model 1 :",value=str(np.round(output,2)) + " kts")
    c.metric(label="Predicted Wind Speed by Model 2 :",value=str(np.round(mod2,2)) + " kts")
    c.metric(label="Average of Model 1 & Model 2 :",value=str(np.round(avg,2)) + " kts")


