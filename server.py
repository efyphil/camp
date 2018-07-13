import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename
from super_resolve import foo 
from PIL import Image

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), "test2")
app.config['OUT_FOLDER'] = os.path.join(os.getcwd(), "output")
@app.route('/', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        print(filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'],filename)
        file.save(filepath)
        filename2 = foo(filename)
        return redirect(url_for('uploaded_file', filename = filename2))
    return render_template('index.html')

@app.route('/output/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['OUT_FOLDER'] ,
                               filename)
if __name__ == "__main__":
    app.run(debug=True)