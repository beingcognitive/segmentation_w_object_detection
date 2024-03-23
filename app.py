import streamlit as st
from transformers import AutoProcessor, AutoModelForMaskGeneration
from transformers import pipeline
from PIL import Image, ImageOps
# from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt
import torch
import requests
from io import BytesIO

def main():
    st.title("Image Segmentation")
    
    # Load SAM by Facebook
    processor = AutoProcessor.from_pretrained("facebook/sam-vit-huge")
    model = AutoModelForMaskGeneration.from_pretrained("facebook/sam-vit-huge")
    # Load Object Detection
    od_pipe = pipeline("object-detection", "facebook/detr-resnet-50")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    xs_ys = [(2.0, 2.0), (2.5, 2.5)] #, (2.5, 2.0), (2.0, 2.5), (1.5, 1.5)]
    alpha = 20
    width = 600

    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file)
    
        st.subheader("Uploaded Image")
        st.image(raw_image, caption="Uploaded Image", width=width)

        ### STEP 1. Object Detection
        pipeline_output = od_pipe(raw_image)
        
        # Convert the bounding boxes from the pipeline output into the expected format for the SAM processor
        input_boxes_format = [[[b['box']['xmin'], b['box']['ymin']], [b['box']['xmax'], b['box']['ymax']]] for b in pipeline_output]
        labels_format = [b['label'] for b in pipeline_output]
        print(input_boxes_format)
        print(labels_format)

        # Now use these formatted boxes with the processor
        for b, l in zip(input_boxes_format, labels_format):
            with st.spinner('Processing...'):

                st.subheader(f'bounding box : {l}')
                inputs = processor(images=raw_image,
                                   input_boxes=[b],
                                   return_tensors="pt")

                with torch.no_grad():
                    outputs = model(**inputs)

                predicted_masks = processor.image_processor.post_process_masks(
                    outputs.pred_masks,
                    inputs["original_sizes"],
                    inputs["reshaped_input_sizes"]
                )
                predicted_mask = predicted_masks[0]

                for i in range(0, 3):
                    # 2D array (boolean mask)
                    mask = predicted_mask[0][i]
                    int_mask = np.array(mask).astype(int) * 255
                    mask_image = Image.fromarray(int_mask.astype('uint8'), mode='L')

                    # Apply the mask to the image
                    # Convert mask to a 3-channel image if your base image is in RGB
                    mask_image_rgb = ImageOps.colorize(mask_image, (0, 0, 0), (255, 255, 255))
                    final_image = Image.composite(raw_image, Image.new('RGB', raw_image.size, (255,255,255)), mask_image)
                    
                    #display the final image
                    st.image(final_image, caption=f"Masked Image {i+1}", width=width)

        ###
        for (x, y) in xs_ys:
            with st.spinner('Processing...'):

                # Calculate input points
                point_x = raw_image.size[0] // x
                point_y = raw_image.size[1] // y
                input_points = [[[ point_x, point_y ]]]

                # Prepare inputs
                inputs = processor(images=raw_image, input_points=input_points, return_tensors="pt")

                # Generate masks
                with torch.no_grad():
                    outputs = model(**inputs)

                # Post-process masks
                predicted_masks = processor.image_processor.post_process_masks(
                    outputs.pred_masks,
                    inputs["original_sizes"],
                    inputs["reshaped_input_sizes"]
                )

                predicted_mask = predicted_masks[0]

                # Display masked images
                st.subheader(f"Input points : ({1/x},{1/y})")
                for i in range(3):
                    mask = predicted_mask[0][i]
                    int_mask = np.array(mask).astype(int) * 255
                    mask_image = Image.fromarray(int_mask.astype('uint8'), mode='L')

                    ###
                    mask_image_rgb = ImageOps.colorize(mask_image, (0, 0, 0), (255, 255, 255))
                    final_image = Image.composite(raw_image, Image.new('RGB', raw_image.size, (255,255,255)), mask_image)

                    st.image(final_image, caption=f"Masked Image {i+1}", width=width)

if __name__ == "__main__":
    main()
