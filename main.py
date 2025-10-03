import pyautogui
import numpy as np
import cv2
import time
import random
import logging
import pygetwindow as gw
from pynput import keyboard
from threading import Thread
import math
import os
from sklearn.linear_model import LinearRegression
import json

# -------------------【日志与文件配置】-------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
ZONES_FILENAME = "exclusion_zones.txt"
COLOR_FILENAME = "exclusion_zones.txt" # 使用同一个文件来存储颜色，但以不同的方式管理

# -------------------【重要参数配置】-------------------
# 请在这里修改重要参数以调整脚本行为。
# --------------------------------------------------

# 1. 目标颜色与容差
TARGET_COLOR_RGB = (0, 0, 0)      # 默认目标颜色 (R, G, B), 可通过 Q 热键动态修改。
                                   # 脚本启动时会自动从文件中加载上一次的颜色。
TOLERANCE = 4                     # 颜色容差，值越大，识别越宽松。

# 2. 鼠标移动与点击速度
# 移动速度，值越大越慢，越"人性化"
MOUSE_MOVE_DURATION_BASE = 0    # 增加这个值来减慢鼠标移动速度
MOUSE_MOVE_DURATION_RANGE = 0    # 增加这个值来增加移动随机性
# 点击间隔，值越大越慢
BASE_CLICK_INTERVAL = 0    # 增加这个值来减慢点击速度
RANDOM_DELAY_RANGE = 0    # 增加这个值来增加随机性

# 3. 鼠标抖动与偏移
# 鼠标抖动停止阈值，当鼠标移动距离超过该值时，脚本暂停。
SHAKE_STOP_THRESHOLD = 300
# 随机偏移像素，增加"人性化"
RANDOM_OFFSET_PIXELS = 1

# 4. 绘图与优化
USE_SPACEBAR_DRAG = True          # 是否启用空格拖动优化。
PIXEL_ADJACENCY_THRESHOLD = 50   # 判断像素是否连续的距离阈值。

# 5. 窗口与区域配置
TARGET_WINDOW_TITLE = "Paint the world"
CROP_WINDOW_TOP = 185            # 从窗口顶部裁剪掉的高度，用于避开地址栏/书签栏。
EXCLUSION_ZONES = []            # 备用列表, 脚本会优先从 exclusion_zones.txt 加载。

# 6. 热键配置
# 热键已更新：暂停/继续为 'w'，取色为 'q'
HOTKEY_RUN_PAUSE = keyboard.KeyCode.from_char('w')
HOTKEY_PICK_COLOR = keyboard.KeyCode.from_char('q')
HOTKEY_ZONE_CORNER = keyboard.Key.f9
HOTKEY_ZONE_SAVE = keyboard.Key.f10
HOTKEY_TERMINATE = keyboard.KeyCode.from_char('g') # G键，用于终止脚本

# 7. 调试模式
DEBUG_MODE = False                 # 开启后，脚本只移动鼠标但不执行点击。

# -------------------【全局状态变量】-------------------
script_active = False
terminate_script = False # 新增全局变量，用于控制脚本退出
targets_to_click = []
target_window_info = {}
last_mouse_pos = pyautogui.position()
# F9 区域选择状态
zone_capture_step = 0
temp_zone_p1 = (0, 0)
temp_zone_rect = (0, 0, 0, 0)

# 创建键盘控制器用于模拟按键
keyboard_controller = keyboard.Controller()

# -------------------【智能色彩校正】-------------------
class ColorCorrector:
    """通过已知数据训练模型, 用于反算wplace悬停颜色效果"""
    def __init__(self):
        # 悬停颜色 (X) -> 真实颜色 (y)
        hover_colors = np.array([
            [126, 122, 161], [82, 82, 82], [235, 235, 235], [231, 215, 117]
        ])
        real_colors = np.array([
            [74, 66, 132], [0, 0, 0], [255, 255, 255], [249, 221, 59]
        ])
        self.models = {}
        for i, channel in enumerate(['R', 'G', 'B']):
            X = hover_colors[:, i].reshape(-1, 1)
            y = real_colors[:, i]
            model = LinearRegression()
            model.fit(X, y)
            self.models[channel] = model
        logging.info("智能色彩校正模型已初始化。")

    def reverse(self, hover_rgb):
        """根据悬停颜色预测真实颜色"""
        r, g, b = hover_rgb
        pred_r = self.models['R'].predict(np.array([[r]]))[0]
        pred_g = self.models['G'].predict(np.array([[g]]))[0]
        pred_b = self.models['B'].predict(np.array([[b]]))[0]
        # 约束到 0-255 范围并取整
        real_rgb = (
            int(np.clip(pred_r, 0, 255)),
            int(np.clip(pred_g, 0, 255)),
            int(np.clip(pred_b, 0, 255))
        )
        return real_rgb

color_corrector = ColorCorrector()

# -------------------【文件与区域管理】-------------------
def load_exclusion_zones():
    """
    从文件中加载忽略区域和目标颜色。
    如果文件不存在，则自动创建。
    """
    global EXCLUSION_ZONES, TARGET_COLOR_RGB
    if os.path.exists(ZONES_FILENAME):
        try:
            with open(ZONES_FILENAME, 'r') as f:
                content = f.read().strip()
                if not content:
                    logging.info(f"{ZONES_FILENAME} 文件为空。")
                    return

                lines = content.split('\n')
                # 尝试加载颜色
                try:
                    last_line = lines[-1].strip()
                    if last_line.startswith('(') and last_line.endswith(')'):
                        loaded_color = eval(last_line)
                        if isinstance(loaded_color, tuple) and len(loaded_color) == 3:
                            TARGET_COLOR_RGB = loaded_color
                            logging.info(f"成功从 {ZONES_FILENAME} 加载上次的目标颜色: {TARGET_COLOR_RGB}")
                            lines.pop() # 移除颜色行，只处理区域
                except:
                    logging.warning("未找到上次保存的颜色或颜色格式不正确。")

                # 加载区域
                loaded_zones = []
                for line in lines:
                    line = line.strip()
                    if line:
                        try:
                            loaded_zones.append(eval(line))
                        except Exception as e:
                            logging.error(f"跳过无效的区域格式: {line} - {e}")

                EXCLUSION_ZONES = loaded_zones
                logging.info(f"成功从 {ZONES_FILENAME} 加载 {len(loaded_zones)} 个忽略区域。")

        except Exception as e:
            logging.error(f"从 {ZONES_FILENAME} 加载文件失败: {e}")
    else:
        # 如果文件不存在，则创建它
        try:
            with open(ZONES_FILENAME, 'w') as f:
                pass # 创建空文件
            logging.info(f"{ZONES_FILENAME} 未找到，已自动为您创建。")
        except Exception as e:
            logging.error(f"创建 {ZONES_FILENAME} 文件失败: {e}")
        
        logging.info("将使用脚本内配置的忽略区域和默认颜色。")


def save_exclusion_data(zone_rect=None, color_rgb=None):
    """保存忽略区域或目标颜色到文件"""
    # 读取旧数据
    old_zones = []
    old_color = None
    if os.path.exists(ZONES_FILENAME):
        try:
            with open(ZONES_FILENAME, 'r') as f:
                content = f.read().strip()
                if content:
                    lines = content.split('\n')
                    try:
                        last_line = lines[-1].strip()
                        if last_line.startswith('(') and last_line.endswith(')'):
                            old_color = eval(last_line)
                            lines.pop()
                    except:
                        pass
                    old_zones = [eval(line.strip()) for line in lines if line.strip()]
        except Exception as e:
            logging.error(f"读取旧数据失败: {e}")

    # 确定要保存的新数据
    if zone_rect:
        old_zones.append(zone_rect)
        logging.info(f"新忽略区域 {zone_rect} 已添加到缓存。")
    if color_rgb:
        old_color = color_rgb
        logging.info(f"新颜色 {color_rgb} 已添加到缓存。")

    # 写入文件
    try:
        with open(ZONES_FILENAME, 'w') as f:
            for zone in old_zones:
                f.write(f"{zone}\n")
            if old_color:
                f.write(f"{old_color}\n")
        logging.info(f"数据已成功保存到 {ZONES_FILENAME}。")
    except Exception as e:
        logging.error(f"保存数据失败: {e}")
    
    # 重新加载以更新内存
    load_exclusion_zones()

# -------------------【核心绘画逻辑】-------------------
def get_active_window_info():
    """获取并返回当前活动窗口的信息, 用于坐标转换"""
    windows = gw.getWindowsWithTitle(TARGET_WINDOW_TITLE)
    if not windows: return None
    window = windows[0]
    # pygetwindow.top 是指窗口标题栏顶部，所以这里需要加上它来获得窗口内容区域的顶部
    win_x = max(0, window.left)
    win_y = max(0, window.top)
    return {'left': win_x, 'top': win_y + CROP_WINDOW_TOP, 'width': window.width, 'height': window.height - CROP_WINDOW_TOP}

def group_targets_for_drag(targets):
    """将相邻的像素点分组, 用于空格拖动优化"""
    if not targets: return []
    
    groups = []
    current_group = [targets[0]]
    
    for i in range(1, len(targets)):
        px, py = targets[i]
        last_x, last_y = current_group[-1]
        dist = math.sqrt((px - last_x)**2 + (py - last_y)**2)
        if dist < PIXEL_ADJACENCY_THRESHOLD:
            current_group.append(targets[i])
        else:
            groups.append(current_group)
            current_group = [targets[i]]
    groups.append(current_group)
    
    logging.info(f"优化分析: {len(targets)}个目标点被分成了 {len(groups)} 组笔画。")
    return groups

def scan_for_targets():
    """扫描窗口, 找出所有目标并填充任务列表"""
    global targets_to_click, target_window_info
    
    logging.info("------------ 开始扫描新任务 ------------")
    win_info = get_active_window_info()
    if not win_info:
        logging.warning(f"未找到目标窗口 '{TARGET_WINDOW_TITLE}'。")
        return False
    
    target_window_info = win_info
    try:
        # 确保截图区域是有效的
        screenshot_region = (win_info['left'], win_info['top'], win_info['width'], win_info['height'])
        screenshot = pyautogui.screenshot(region=screenshot_region)
        screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logging.error(f"截图失败: {e}。")
        return False
    
    global TARGET_COLOR_RGB
    target_color_bgr = (TARGET_COLOR_RGB[2], TARGET_COLOR_RGB[1], TARGET_COLOR_RGB[0])
    lower = np.array([max(0, c - TOLERANCE) for c in target_color_bgr])
    upper = np.array([min(255, c + TOLERANCE) for c in target_color_bgr])
    mask = cv2.inRange(screenshot_cv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logging.info("扫描完成: 未找到任何指定颜色的区域。")
        return True
    
    valid_contours = [c for c in contours if cv2.boundingRect(c)[2]>1 and cv2.boundingRect(c)[3]>1 and not is_in_exclusion_zone(*cv2.boundingRect(c))]
    if not valid_contours:
        logging.info("扫描完成: 所有色块均在忽略区域内。")
        return True
        
    areas = [cv2.contourArea(c) for c in valid_contours]
    median_area = np.median(areas)
    min_area, max_area = median_area * 0.5, median_area * 1.5
    
    temp_targets = []
    for c in valid_contours:
        x, y, w, h = cv2.boundingRect(c)
        if min_area <= w * h <= max_area:
            temp_targets.append((x + w // 2, y + h // 2))

    if not temp_targets:
        logging.info("扫描完成: 未找到满足尺寸条件的标记。")
        return True
    
    temp_targets.sort(key=lambda p: (p[1], p[0]))
    targets_to_click = temp_targets
    logging.info(f"扫描完成: 找到 {len(targets_to_click)} 个有效目标, 已生成任务列表。")
    return True

def is_in_exclusion_zone(x, y, w, h):
    rect_x2, rect_y2 = x + w, y + h
    for zx, zy, zw, zh in EXCLUSION_ZONES:
        zone_x2, zone_y2 = zx + zw, zy + zh
        if not (rect_x2 < zx or x > zone_x2 or rect_y2 < zy or y > zone_y2):
            return True
    return False

# -------------------【主循环与热键控制】-------------------
def pause_script(reason=""):
    """统一的暂停脚本函数"""
    global script_active, targets_to_click
    if script_active:
        script_active = False
        targets_to_click = [] # 暂停时清空任务列表
        logging.info(f"========== 脚本已 [暂停] (原因: {reason}) ==========")

def on_press(key):
    global script_active, terminate_script, zone_capture_step, temp_zone_p1, TARGET_COLOR_RGB, last_mouse_pos

    # 'G' 键: 退出脚本
    if key == HOTKEY_TERMINATE:
        logging.info("========== 检测到 'G' 键, 正在终止脚本... ==========")
        terminate_script = True
        return False # 返回False来停止键盘监听器

    # 'w' 键: 启动/暂停
    if key == HOTKEY_RUN_PAUSE:
        if script_active:
            pause_script("'w'键按下")
        else:
            script_active = True
            logging.info("========== 已按下 'w', 脚本已 [启动/继续] ==========")
            last_mouse_pos = pyautogui.position()
        return

    # 在任何功能键按下时, 如果脚本在运行, 则先暂停
    if script_active:
        if key in [HOTKEY_PICK_COLOR, HOTKEY_ZONE_CORNER, HOTKEY_ZONE_SAVE]:
            pause_script(f"功能键 {key} 按下")
    
    # 'q' 键: 拾取颜色
    if key == HOTKEY_PICK_COLOR:
        pos = pyautogui.position()
        hover_color = pyautogui.pixel(pos.x, pos.y)
        real_color = color_corrector.reverse(hover_color)
        TARGET_COLOR_RGB = real_color
        save_exclusion_data(color_rgb=real_color) # 立即保存新颜色
        logging.info(f"颜色选择: 鼠标位置 {pos}, 悬停色 {hover_color}, 预测真实颜色 -> {real_color}")
        return

    # F9/F10: 区域选择
    if key == HOTKEY_ZONE_CORNER:
        if zone_capture_step == 0:
            temp_zone_p1 = pyautogui.position()
            zone_capture_step = 1
            logging.info(f"区域选择(1/2): 已记录左上角坐标 {temp_zone_p1}。请移动鼠标到右下角再按 F9。")
        elif zone_capture_step == 1:
            p2 = pyautogui.position()
            win_info = get_active_window_info()
            if not win_info:
                logging.warning("区域选择失败: 未找到目标窗口, 无法计算相对坐标。")
                zone_capture_step = 0
                return
            # 转换为相对于窗口内容区的坐标
            x1 = min(temp_zone_p1.x, p2.x) - win_info['left']
            y1 = min(temp_zone_p1.y, p2.y) - win_info['top']
            x2 = max(temp_zone_p1.x, p2.x) - win_info['left']
            y2 = max(temp_zone_p1.y, p2.y) - win_info['top']
            w, h = x2 - x1, y2 - y1
            
            global temp_zone_rect
            temp_zone_rect = (int(x1), int(y1), int(w), int(h))
            zone_capture_step = 2
            logging.info(f"区域选择(2/2): 右下角 {p2}。计算出相对区域: {temp_zone_rect}。请按 F10 保存。")
    elif key == HOTKEY_ZONE_SAVE:
        if zone_capture_step == 2:
            save_exclusion_data(zone_rect=temp_zone_rect)
            zone_capture_step = 0
        else:
            logging.warning("区域保存失败: 请先按两次 F9 来定义一个区域。")
    elif zone_capture_step > 0:
        logging.info("区域选择已取消。")
        zone_capture_step = 0

def start_keyboard_listener():
    # 在这里，监听器会因为 on_press 中对 'g' 的 `return False` 而停止
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def main():
    load_exclusion_zones()
    logging.info(f"“究极人性化”脚本 v6 已启动。调试模式: {'开启' if DEBUG_MODE else '关闭'}")
    logging.info(f"W:启动/暂停 | Q:取色 | F9-F9-F10:定义并保存忽略区域 | G:强制退出脚本")
    
    global last_mouse_pos
    last_mouse_pos = pyautogui.position()

    # 主循环将一直运行，直到 terminate_script 被设置为 True
    while not terminate_script:
        try:
            if not script_active:
                time.sleep(0.1)
                continue

            if not targets_to_click:
                if not scan_for_targets() or not targets_to_click:
                    pause_script("扫描结束或失败")
                    continue
            
            # --- 晃动检测 ---
            current_pos = pyautogui.position()
            if math.sqrt((current_pos.x - last_mouse_pos.x)**2 + (current_pos.y - last_mouse_pos.y)**2) > SHAKE_STOP_THRESHOLD:
                pause_script("鼠标晃动")
                continue

            # --- 任务执行 ---
            task_groups = group_targets_for_drag(targets_to_click) if USE_SPACEBAR_DRAG else [[t] for t in targets_to_click]
            total_tasks = len(targets_to_click)
            tasks_done = 0
            
            for group in task_groups:
                if not script_active or terminate_script: break # 检查在处理组之间是否被暂停或终止
                
                # 移动到组的第一个点
                start_pos = group[0]
                final_x = target_window_info['left'] + start_pos[0] + random.randint(-RANDOM_OFFSET_PIXELS, RANDOM_OFFSET_PIXELS)
                final_y = target_window_info['top'] + start_pos[1] + random.randint(-RANDOM_OFFSET_PIXELS, RANDOM_OFFSET_PIXELS)
                pyautogui.moveTo(final_x, final_y, duration=max(0.05, MOUSE_MOVE_DURATION_BASE + random.uniform(-MOUSE_MOVE_DURATION_RANGE, MOUSE_MOVE_DURATION_RANGE)), tween=pyautogui.easeOutQuad)
                
                if len(group) == 1: # 单点, 直接点击
                    if not DEBUG_MODE: pyautogui.click()
                    logging.info(f"执行 {tasks_done+1}/{total_tasks} | 单点:({final_x},{final_y}) {'(调试)' if DEBUG_MODE else ''}")
                    # 【修复】单点点击后立即应用延迟
                    time.sleep(max(0, BASE_CLICK_INTERVAL + random.uniform(-RANDOM_DELAY_RANGE, RANDOM_DELAY_RANGE)))
                else: # 多点, 按住空格拖动
                    logging.info(f"执行 {tasks_done+1}-{tasks_done+len(group)}/{total_tasks} | 拖动画笔, 共 {len(group)} 点...")
                    
                    if not DEBUG_MODE: 
                        # 使用pynput的键盘控制器按下空格键
                        keyboard_controller.press(keyboard.Key.space)
                        time.sleep(0.2)  # 增加延迟确保空格键被识别
                    
                    try:
                        for i in range(1, len(group)):
                            if not script_active or terminate_script: break
                            point = group[i]
                            final_x = target_window_info['left'] + point[0] + random.randint(-RANDOM_OFFSET_PIXELS, RANDOM_OFFSET_PIXELS)
                            final_y = target_window_info['top'] + point[1] + random.randint(-RANDOM_OFFSET_PIXELS, RANDOM_OFFSET_PIXELS)
                            pyautogui.moveTo(final_x, final_y, duration=0.01)
                            # 拖动时每个点之间也应用延迟
                            time.sleep(max(0, BASE_CLICK_INTERVAL + random.uniform(-RANDOM_DELAY_RANGE, RANDOM_DELAY_RANGE)))
                    finally:
                        # 确保空格键被释放
                        if not DEBUG_MODE: 
                            time.sleep(0.1)  # 拖动结束前短暂延迟
                            keyboard_controller.release(keyboard.Key.space)
                    
                    # 拖动完成后应用延迟
                    time.sleep(max(0, BASE_CLICK_INTERVAL + random.uniform(-RANDOM_DELAY_RANGE, RANDOM_DELAY_RANGE)))
                
                tasks_done += len(group)
                last_mouse_pos = pyautogui.position()
                # 【注意】移除了原来在组末尾的 sleep，因为现在每个操作后都有独立的延迟
            
            pause_script("本轮任务完成")

        except KeyboardInterrupt:
            logging.info("检测到 Ctrl+C, 脚本已退出。")
            break
        except Exception as e:
            logging.error(f"主循环发生严重错误: {e}")
            pause_script("异常错误")
            time.sleep(2)

if __name__ == "__main__":
    listener_thread = Thread(target=start_keyboard_listener, daemon=True)
    listener_thread.start()
    main()
    logging.info("脚本已安全退出。")
