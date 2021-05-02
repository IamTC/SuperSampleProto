import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import *
import PIL.Image
import time
import glob
from functools import cmp_to_key
import rsslib
import binascii
import pygame

import os
import io

from array import array


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


# initializing glfw library
if not glfw.init():
    raise Exception("glfw can not be initialized!")

# creating the window
glfw.window_hint(glfw.DOUBLEBUFFER, GL_FALSE)
window = glfw.create_window(
    1280, 900, "Realtime Supersampling Demo", None, None)

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

vertices = [
    -1, -1,  0.5,  1.0, 0.0, 0.0,  0.0, 0.0,
    1, -1,  0.5,  0.0, 1.0, 0.0,  1.0, 0.0,
    1,  1,  0.5,  0.0, 0.0, 1.0,  1.0, 1.0,
    -1,  1,  0.5,  1.0, 1.0, 1.0,  0.0, 1.0,

    -1, -1, -0.5,  1.0, 0.0, 0.0,  0.0, 0.0,
    1, -1, -0.5,  0.0, 1.0, 0.0,  1.0, 0.0,
    1,  1, -0.5,  0.0, 0.0, 1.0,  1.0, 1.0,
    -1,  1, -0.5,  1.0, 1.0, 1.0,  0.0, 1.0,

    1, -1, -0.5,  1.0, 0.0, 0.0,  0.0, 0.0,
    1,  1, -0.5,  0.0, 1.0, 0.0,  1.0, 0.0,
    1,  1,  0.5,  0.0, 0.0, 1.0,  1.0, 1.0,
    1, -1,  0.5,  1.0, 1.0, 1.0,  0.0, 1.0,

    -1, 1, -0.5,  1.0, 0.0, 0.0,  0.0, 0.0,
    -1, -1, -0.5,  0.0, 1.0, 0.0,  1.0, 0.0,
    -1, -1,  0.5,  0.0, 0.0, 1.0,  1.0, 1.0,
    -1,  1,  0.5,  1.0, 1.0, 1.0,  0.0, 1.0,

    -1, -1, -0.5,  1.0, 0.0, 0.0,  0.0, 0.0,
    1, -1, -0.5,  0.0, 1.0, 0.0,  1.0, 0.0,
    1, -1,  0.5,  0.0, 0.0, 1.0,  1.0, 1.0,
    -1, -1,  0.5,  1.0, 1.0, 1.0,  0.0, 1.0,

    1,  1, -0.5,  1.0, 0.0, 0.0,  0.0, 0.0,
    -1,  1, -0.5,  0.0, 1.0, 0.0,  1.0, 0.0,
    -1, 1,  0.5,  0.0, 0.0, 1.0,  1.0, 1.0,
    1,  1,  0.5,  1.0, 1.0, 1.0,  0.0, 1.0
]

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

# Vertex Buffer Object for non immediate-mode rendering. 
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


glUseProgram(shader)
glClearColor(0, 0.1, 0.1, 1)
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

rotation_loc = glGetUniformLocation(shader, "rotation")

images = []
for fname in glob.iglob('textures/*'):
    images.append(Image.open(fname))


def get_frame_number(frame):
    frame = frame.filename
    frame = frame.split('\\')[1]
    frame = frame.split('.')[0]

    return int(frame)

pygame.font.init()

def drawText(position, textString):    
    font = pygame.font.Font (None, 64)
    textSurface = font.render(textString, True, (255,255,255,255), (0,0,0,255))     
    textData = pygame.image.tostring(textSurface, "RGBA", True)     
    glRasterPos3d(1,1,0)     
    glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)


images = sorted(images, key=cmp_to_key(
    lambda img1, img2: get_frame_number(img1) - get_frame_number(img2)))

target_fps = 20.0

model_path = "./model/model_SeparableConv4x.h5"
rss = rsslib.RSS(image_size_h=900, image_size_w=1600,
                 batch_size=8, upscale_factor=4, channels=1)
rss.load_model(model_path)

# the main application loop
lastTime = glfw.get_time()
nbFrames = 0
frame_idx = 0
frame_time = 0
while not glfw.window_should_close(window): 
    currentTime = glfw.get_time()
    nbFrames += 1
    out = images[frame_idx].transpose(Image.FLIP_TOP_BOTTOM)
    t0 =glfw.get_time()
    out = rss.upscale_image(mode="yuv", image=out)
    t1 = glfw.get_time()
    print("Upscale Time: " , (t1-t0) * 1000)
    image_arr_data = img_to_array(out)
    img_data = out
    img_data = out.convert("RGBA").tobytes()

    # tf.keras.preprocessing.image.save_img(
    #     './ResultNoSR/' + str(frame_idx) + '.png', image_arr_data, data_format=None, scale=True
    # )

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, out.width, out.height,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

    glfw.poll_events()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    rot_x = pyrr.Matrix44.from_x_rotation(0)
    rot_y = pyrr.Matrix44.from_y_rotation(0)

    glUniformMatrix4fv(rotation_loc, 1, GL_FALSE,
                       pyrr.matrix44.multiply(rot_x, rot_y))

    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    # glfw.swap_buffers(window)

    lastTime = glfw.get_time()

    frame_time = (lastTime - currentTime) * 1000

    glDisable(GL_TEXTURE_2D)
    glEnable(GL_COLOR_MATERIAL)

    if(frame_time >= target_fps):
        frame_idx += 1
        # print("Frame Time:", frame_time)        
        
    glFlush()
    glFinish()

# terminate glfw, free up allocated resources
glfw.terminate()
