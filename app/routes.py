from flask import Blueprint, render_template, request, send_file
from .translation_and_image import generate_image, get_translation
import io

main_routes = Blueprint('main', __name__)

@main_routes.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form.get('text')
        dest_lang = request.form.get('language')

        # Translate the text
        translated_text = get_translation(text, dest_lang)

        # Generate the image based on translated text
        image = generate_image(translated_text)

        # Save image to a byte buffer and send as response
        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    return render_template('index.html')

