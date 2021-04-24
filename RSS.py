import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
from pygame import *
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import *
import PIL.Image
import time
import glob
from functools import cmp_to_key

model_path = "model/model_SeparableConv.h5"

vertex_src = """
# version 330
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;
layout(location = 2) in vec2 a_texture;
uniform mat4 rotation;
out vec3 v_color;
out vec2 v_texture;
void main()
{
    gl_Position = rotation * vec4(a_position, 1.0);
    v_color = a_color;
    v_texture = a_texture;
    
    //v_texture = 1 - a_texture;                      // Flips the texture vertically and horizontally
    //v_texture = vec2(a_texture.s, 1 - a_texture.t); // Flips the texture vertically
}
"""

fragment_src = """
# version 330
in vec3 v_color;
in vec2 v_texture;
out vec4 out_color;
uniform sampler2D s_texture;
void main()
{
    out_color = texture(s_texture, v_texture); // * vec4(v_color, 1.0f);
}
"""

# glfw callback functions


def window_resize(window, width, height):
    glViewport(0, 0, width, height)


def upscale_image(model, img):
    """Predict the result based on input image and restore the image as RGB."""
    ycbcr = img.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y = img_to_array(y)
    y = y.astype("float32") / 255.0

    input = np.expand_dims(y, axis=0)
    t0 = time.perf_counter()
    out = model.predict(input)
    t1 = time.perf_counter()
    print("Upscale time", t1-t0)

    out_img_y = out[0]
    out_img_y *= 255.0

    # Restore the image in RGB color space.
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = out_img_y.reshape(
        (np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
    out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
    out_img_cb = cb.resize(out_img_y.size, PIL.Image.NEAREST)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.NEAREST)
    out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert(
        "RGB"
    )
    return out_img


def upscale_image_rgb(model, img):
    """Predict the result based on input image and restore the image as RGB."""
    # ycbcr = img.convert("YCbCr")
    # y, cb, cr = ycbcr.split()
    y = img_to_array(img)
    y = y.astype("float32") / 255.0

    input = np.expand_dims(y, axis=0)
    out = model.predict(input)

    out_img_y = out[0]
    out_img_y *= 255.0
    out_img_y = np.uint8(out_img_y)
    out_img_y = Image.fromarray(out_img_y)

    # Restore the image in RGB color space.
    # out_img_y = out_img_y.clip(0, 255)
    # out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
    # out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
    # out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
    # out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)

    return out_img_y


def get_lowres_image(img):
    """Return low-resolution image to use as model input."""
    return img.resize(
        (img.size[0] // 3, img.size[1] // 3),
        PIL.Image.BICUBIC,
    )


# initializing glfw library
if not glfw.init():
    raise Exception("glfw can not be initialized!")

# creating the window
glfw.window_hint(glfw.DOUBLEBUFFER, GL_FALSE)
window = glfw.create_window(1280, 720, "Realtime Supersampling Demo", None, None)

# check if window was created
if not window:
    glfw.terminate()
    raise Exception("glfw window can not be created!")

# set window's position
glfw.set_window_pos(window, 10, 50)

# set the callback function for window resize
glfw.set_window_size_callback(window, window_resize)

# make the context current
glfw.make_context_current(window)

vertices = [-0.5, -0.5,  0.5,  1.0, 0.0, 0.0,  0.0, 0.0,
            0.5, -0.5,  0.5,  0.0, 1.0, 0.0,  1.0, 0.0,
            0.5,  0.5,  0.5,  0.0, 0.0, 1.0,  1.0, 1.0,
            -0.5,  0.5,  0.5,  1.0, 1.0, 1.0,  0.0, 1.0,

            -0.5, -0.5, -0.5,  1.0, 0.0, 0.0,  0.0, 0.0,
            0.5, -0.5, -0.5,  0.0, 1.0, 0.0,  1.0, 0.0,
            0.5,  0.5, -0.5,  0.0, 0.0, 1.0,  1.0, 1.0,
            -0.5,  0.5, -0.5,  1.0, 1.0, 1.0,  0.0, 1.0,

            0.5, -0.5, -0.5,  1.0, 0.0, 0.0,  0.0, 0.0,
            0.5,  0.5, -0.5,  0.0, 1.0, 0.0,  1.0, 0.0,
            0.5,  0.5,  0.5,  0.0, 0.0, 1.0,  1.0, 1.0,
            0.5, -0.5,  0.5,  1.0, 1.0, 1.0,  0.0, 1.0,

            -0.5,  0.5, -0.5,  1.0, 0.0, 0.0,  0.0, 0.0,
            -0.5, -0.5, -0.5,  0.0, 1.0, 0.0,  1.0, 0.0,
            -0.5, -0.5,  0.5,  0.0, 0.0, 1.0,  1.0, 1.0,
            -0.5,  0.5,  0.5,  1.0, 1.0, 1.0,  0.0, 1.0,

            -0.5, -0.5, -0.5,  1.0, 0.0, 0.0,  0.0, 0.0,
            0.5, -0.5, -0.5,  0.0, 1.0, 0.0,  1.0, 0.0,
            0.5, -0.5,  0.5,  0.0, 0.0, 1.0,  1.0, 1.0,
            -0.5, -0.5,  0.5,  1.0, 1.0, 1.0,  0.0, 1.0,

            0.5,  0.5, -0.5,  1.0, 0.0, 0.0,  0.0, 0.0,
            -0.5,  0.5, -0.5,  0.0, 1.0, 0.0,  1.0, 0.0,
            -0.5,  0.5,  0.5,  0.0, 0.0, 1.0,  1.0, 1.0,
            0.5,  0.5,  0.5,  1.0, 1.0, 1.0,  0.0, 1.0]

indices = [0,  1,  2,  2,  3,  0,
           4,  5,  6,  6,  7,  4,
           8,  9, 10, 10, 11,  8,
           12, 13, 14, 14, 15, 12,
           16, 17, 18, 18, 19, 16,
           20, 21, 22, 22, 23, 20]

vertices = np.array(vertices, dtype=np.float32)
indices = np.array(indices, dtype=np.uint32)

shader = compileProgram(compileShader(
    vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))

# Vertex Buffer Object
VBO = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

# Element Buffer Object
EBO = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                      vertices.itemsize * 8, ctypes.c_void_p(0))

glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
                      vertices.itemsize * 8, ctypes.c_void_p(12))

glEnableVertexAttribArray(2)
glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE,
                      vertices.itemsize * 8, ctypes.c_void_p(24))

texture = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, texture)

# Set the texture wrapping parameters
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
# Set texture filtering parameters
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

new_model = tf.keras.models.load_model(model_path)
new_model.summary()


glUseProgram(shader)
glClearColor(0, 0.1, 0.1, 1)
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

rotation_loc = glGetUniformLocation(shader, "rotation")

# load image
image = Image.open("textures/frame (1).png")
image = image.transpose(Image.FLIP_TOP_BOTTOM)
# img_data = image.convert("RGBA").tobytes()
# img_data = np.array(image.getdata(), np.uint8) # second way of getting the raw image data

images = []
for fname in glob.iglob('textures/*'):
    images.append(Image.open(fname))


def get_frame_number(frame):
    frame = frame.filename
    frame = frame.split('(')[1]
    frame = frame.split('.')[0]
    frame = frame.replace(')', '')

    return int(frame)


images = sorted(images, key=cmp_to_key(
    lambda img1, img2: get_frame_number(img1) - get_frame_number(img2)))

target_fps = 20.0

# the main application loop
lastTime = glfw.get_time()
nbFrames = 0
frame_idx = 0
while not glfw.window_should_close(window):
    currentTime = glfw.get_time()
    nbFrames += 1
    # if currentTime - lastTime >= 1.0:
    #     # If last prinf() was more than 1 sec ago
    #     # printf and reset timer
    #     print("%f ms/frame\n" % (1000.0/nbFrames))
    #     nbFrames = 0
    #     lastTime += 1.0

    if currentTime - lastTime >= 1/target_fps:
        # If last prinf() was more than 1 sec ago
        # printf and reset timer
        print("%f ms/frame\n" % (1000.0/nbFrames))
        nbFrames = 0
        lastTime += 1/target_fps
        frame_idx += 1
        if(frame_idx == len(images)):
            frame_idx = 0

    out = get_lowres_image(images[frame_idx].transpose(Image.FLIP_TOP_BOTTOM))
    out = upscale_image(new_model, out)
    img_data = out
    img_data = out.convert("RGBA").tobytes()
    # img_data = np.array(image.getdata(), np.uint8)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, out.width, out.height,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

    # tf.keras.preprocessing.image.save_img(
    #     './Result/out', img_data, data_format=None, file_format='png', scale=True
    # )
    # img_data = np.array(image,np.uint8 )

    glfw.poll_events()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # rot_x = pyrr.Matrix44.from_x_rotation(0.5 * glfw.get_time())
    # rot_y = pyrr.Matrix44.from_y_rotation(0.8 * glfw.get_time())

    rot_x = pyrr.Matrix44.from_x_rotation(0)
    rot_y = pyrr.Matrix44.from_y_rotation(0)

    glUniformMatrix4fv(rotation_loc, 1, GL_FALSE,
                       pyrr.matrix44.multiply(rot_x, rot_y))

    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    # glfw.swap_buffers(window)
    glFlush()

# terminate glfw, free up allocated resources
glfw.terminate()
