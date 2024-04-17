from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import base64
import torch
import torchvision.transforms as transforms
from PIL import Image

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/req', methods=['POST'])
def modify_image():
  try:
    data = request.get_json()
    image_data = data['image']
    rgb_value = tuple(eval(data['rgb']))
    print(rgb_value)

    # Decode image data from base64
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    image.save("image.png")
    # Modify image based on RGB value (example: adjust brightness)
    image = image.convert('RGB')

    def load_deeplabv3():
        """Loads a DeepLabV3+ model from a PyTorch checkpoint (.pth file).

        Args:
            model_path (str): Path to the model checkpoint file.
            num_classes (int, optional): Number of output classes (default: 21).
            output_stride (int, optional): Output stride of the model (default: 16).

        Returns:
            torch.nn.Module: The loaded DeepLabV3+ model.
        """


        model = torch.hub.load('pytorch/vision:v0.11.0', 'deeplabv3_resnet101', pretrained=True)
        model.eval()
        return model


    model = load_deeplabv3()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette

    colors = (colors % 255).numpy().astype("uint8")


    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(image.size)
    r.putpalette(colors)

    def apply_color_tone_rgba(img, color_value, r, background, intensity=0.3):
        """Applies a color tone to the image by adjusting channel values and alpha.

        Args:
            img: The PIL Image object to be filtered.
            color_value: A tuple representing the desired color with alpha (e.g., (255, 0, 0, 128) for red with 50% alpha).
            intensity: A value (0.0 to 1.0) controlling the strength of the color filter (default 0.5).

        Returns:
            A new PIL Image object with the color filter applied.
        """
        if r.mode not in ("RGB", "RGBA"):
            r = r.convert("RGBA")  

        pixels = r.load()

        new_img = img.convert('RGBA')  # Ensure RGBA mode

        new_pixels = new_img.load()

        for y in range(new_img.height):
            for x in range(new_img.width):
                if background:
                    if pixels[x, y] == (0, 0, 0, 255):
                        r, g, b, a = new_pixels[x, y]

                        # Adjust each channel based on the provided color_value and intensity
                        new_r = int(r + (color_value[0] - r) * intensity)
                        new_g = int(g + (color_value[1] - g) * intensity)
                        new_b = int(b + (color_value[2] - b) * intensity)

                        # Apply alpha value from the provided color_value
                        new_a = color_value[3]

                        new_pixels[x, y] = (new_r, new_g, new_b, new_a)
                else:
                    if pixels[x, y] != (0, 0, 0, 255):
                        r, g, b, a = new_pixels[x, y]

                        # Adjust each channel based on the provided color_value and intensity
                        new_r = int(r + (color_value[0] - r) * intensity)
                        new_g = int(g + (color_value[1] - g) * intensity)
                        new_b = int(b + (color_value[2] - b) * intensity)

                        # Apply alpha value from the provided color_value
                        new_a = color_value[3]

                        new_pixels[x, y] = (new_r, new_g, new_b, new_a)
        
    

        return new_img

    image = apply_color_tone_rgba(image, rgb_value, r, True)

    image.save("aunita.png")

    # Encode modified image back to base64
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    modified_image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return jsonify({'modified_image': modified_image_data})
  except Exception as e:
    return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)