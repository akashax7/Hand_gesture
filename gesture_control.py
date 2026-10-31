 
import pyautogui

pyautogui.FAILSAFE = True

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

previous_x = SCREEN_WIDTH // 2
previous_y = SCREEN_HEIGHT // 2


def move_mouse(x, y, smoothness=5):

    global previous_x, previous_y

    target_x = int(x * SCREEN_WIDTH)
    target_y = int(y * SCREEN_HEIGHT)

    target_x = max(0, min(target_x, SCREEN_WIDTH - 1))
    target_y = max(0, min(target_y, SCREEN_HEIGHT - 1))

    current_x = previous_x + (target_x - previous_x) / smoothness
    current_y = previous_y + (target_y - previous_y) / smoothness

    pyautogui.moveTo(int(current_x), int(current_y))

    previous_x = current_x
    previous_y = current_y


def left_click():
    pyautogui.click(button="left")


def right_click():
    pyautogui.click(button="right")


def scroll_mouse(amount):
    pyautogui.scroll(amount)
