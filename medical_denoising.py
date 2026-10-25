# ======================================================================
# Medical Image Denoising using Gradient Descent
# Name       : Ashutosh Kumar
# Enrollment : 24014803125        Section : I12
# Course     : Computational Methods Lab, MAIT
#
# Language   : Python 3  (plain loops and lists - no NumPy, no OpenCV,
#              no Matplotlib). Pillow is used ONLY to open / save pictures.
#
# IDEA:  Noise = lots of tiny random jumps between neighbouring pixels.
#        So we define an "error" (energy) that becomes big when
#          (a) the new image is far from the noisy image, or
#          (b) the new image has big jumps between neighbours.
#        Then we use Gradient Descent to make this error small:
#
#              X_new = X_old - learning_rate * gradient
# ======================================================================

import random
import math
from PIL import Image, ImageDraw

# ---------------------- settings you can change -----------------------
IMAGE_FILE = "xray.jpg"
SIZE = 128            # image is shrunk to SIZE x SIZE so pure Python is fast
NOISE = 0.10          # how strong the Gaussian noise is (0 to 1 scale)
LEARNING_RATE = 0.1   # alpha  (step size)
LAMBDA = 3.0          # how strongly we punish jumps between neighbours
ITERATIONS = 100

# Sobel kernels (divided by 8 so they behave like real derivatives)
SOBEL_X = [[-1, 0, 1],
           [-2, 0, 2],
           [-1, 0, 1]]
SOBEL_Y = [[-1, -2, -1],
           [ 0,  0,  0],
           [ 1,  2,  1]]


# ----------------------------- STEP 1 ---------------------------------
def load_image():
    """Open the X-ray, make it grayscale, shrink it, return numbers 0..1."""
    try:
        img = Image.open(IMAGE_FILE).convert("L").resize((SIZE, SIZE))
    except FileNotFoundError:
        print("xray.jpg not found -> using a simple test picture instead")
        img = Image.new("L", (SIZE, SIZE), 30)
        d = ImageDraw.Draw(img)
        d.ellipse((14, 14, 114, 114), fill=130)
        d.rectangle((70, 60, 100, 95), fill=220)
    pixels = list(img.tobytes())
    return [[pixels[r * SIZE + c] / 255 for c in range(SIZE)] for r in range(SIZE)]


# ----------------------------- STEP 2 ---------------------------------
def add_noise(img):
    """Add Gaussian noise to every pixel (keep values inside 0..1)."""
    random.seed(1)
    noisy = []
    for row in img:
        new_row = []
        for p in row:
            p = p + random.gauss(0, NOISE)
            new_row.append(min(1, max(0, p)))
        noisy.append(new_row)
    return noisy


# ----------------------------- STEP 3 ---------------------------------
def sobel(img):
    """Gradient of the image using Sobel. Returns gx, gy (same size)."""
    gx = [[0.0] * SIZE for _ in range(SIZE)]
    gy = [[0.0] * SIZE for _ in range(SIZE)]
    for r in range(1, SIZE - 1):
        for c in range(1, SIZE - 1):
            sx = 0.0
            sy = 0.0
            for i in range(3):
                for j in range(3):
                    p = img[r + i - 1][c + j - 1]
                    sx += SOBEL_X[i][j] * p
                    sy += SOBEL_Y[i][j] * p
            gx[r][c] = sx / 8
            gy[r][c] = sy / 8
    return gx, gy


def find_gradient(X, Y):
    """
    Energy  E = 1/2 * sum (X - Y)^2  +  LAMBDA/2 * sum (gx^2 + gy^2)
    Its gradient for every pixel is:
        (X - Y)  +  LAMBDA * (Sobel pushed back to neighbours)
    """
    gx, gy = sobel(X)

    # part 1: stay close to the noisy image
    grad = [[X[r][c] - Y[r][c] for c in range(SIZE)] for r in range(SIZE)]

    # part 2: punish big jumps (each pixel's gx, gy affects its 3x3 neighbours)
    for r in range(1, SIZE - 1):
        for c in range(1, SIZE - 1):
            for i in range(3):
                for j in range(3):
                    grad[r + i - 1][c + j - 1] += LAMBDA * (
                        SOBEL_X[i][j] * gx[r][c] + SOBEL_Y[i][j] * gy[r][c]) / 8
    return grad


# ----------------------------- STEP 4 ---------------------------------
def gradient_descent(Y):
    """Repeat:  X_new = X_old - learning_rate * gradient"""
    X = [row[:] for row in Y]          # start from the noisy image
    loss_list = []
    for k in range(ITERATIONS):
        grad = find_gradient(X, Y)

        # loss = average of |gradient|  (goes to 0 when we reach the minimum)
        total = 0.0
        for r in range(SIZE):
            for c in range(SIZE):
                total += abs(grad[r][c])
        loss_list.append(total / (SIZE * SIZE))

        # the update rule
        for r in range(SIZE):
            for c in range(SIZE):
                X[r][c] = X[r][c] - LEARNING_RATE * grad[r][c]

        if (k + 1) % 10 == 0:
            print("iteration", k + 1, " loss =", "%.2e" % loss_list[-1])
    return X, loss_list


def psnr(A, B):
    """Quality in dB (higher = closer to the clean image)."""
    mse = 0.0
    for r in range(SIZE):
        for c in range(SIZE):
            mse += (A[r][c] - B[r][c]) ** 2
    mse = mse / (SIZE * SIZE)
    return 10 * math.log10(1 / mse)


# ----------------------------- STEP 5 ---------------------------------
def to_picture(img):
    flat = [int(min(1, max(0, p)) * 255) for row in img for p in row]
    pic = Image.new("L", (SIZE, SIZE))
    pic.putdata(flat)
    return pic.resize((300, 300))


def draw_loss_graph(draw, losses, x0, y0, w, h):
    """Draw the Loss-vs-Iteration graph by hand using lines."""
    draw.rectangle((x0, y0, x0 + w, y0 + h), outline="black")
    top = max(losses)
    points = []
    for k in range(len(losses)):
        x = x0 + k * w / (len(losses) - 1)
        y = y0 + h - losses[k] / top * h
        points.append((x, y))
    draw.line(points, fill="red", width=2)
    draw.text((x0, y0 + h + 6), "Iterations  (0 to " + str(len(losses)) + ")", fill="black")
    draw.text((x0 + 4, y0 + 4), "max loss = " + "%.4f" % top, fill="black")
    draw.text((x0 + w - 130, y0 + h - 16), "final = " + "%.1e" % losses[-1], fill="black")


def save_results(clean, noisy, denoised, losses):
    canvas = Image.new("RGB", (960, 400), "white")
    d = ImageDraw.Draw(canvas)
    canvas.paste(to_picture(noisy).convert("RGB"), (20, 50))
    canvas.paste(to_picture(denoised).convert("RGB"), (340, 50))
    d.text((20, 15), "1) Noisy image  PSNR = " + str(round(psnr(clean, noisy), 2)) + " dB", fill="black")
    d.text((340, 15), "2) Denoised image  PSNR = " + str(round(psnr(clean, denoised), 2)) + " dB", fill="black")
    d.text((670, 15), "3) Loss vs Iterations", fill="black")
    draw_loss_graph(d, losses, 670, 50, 270, 300)
    canvas.save("medical_denoising_results.png")


# ------------------------------ MAIN ----------------------------------
if __name__ == "__main__":
    clean = load_image()
    noisy = add_noise(clean)
    denoised, losses = gradient_descent(noisy)

    print("PSNR noisy    :", round(psnr(clean, noisy), 2), "dB")
    print("PSNR denoised :", round(psnr(clean, denoised), 2), "dB")
    save_results(clean, noisy, denoised, losses)
    print("Saved medical_denoising_results.png")
