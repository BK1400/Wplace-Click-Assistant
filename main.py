import pyautogui
import numpy as np
import cv2
import time
import logging
import pygetwindow as gw
from pynput import keyboard
from threading import Thread
import math
import os
from datetime import datetime

# -------------------【日志与文件配置】-------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# -------------------【重要参数配置】-------------------
# 请在这里修改重要参数以调整脚本行为。
# --------------------------------------------------

# 1. 目标颜色与容差
TARGET_COLOR_RGB = (0, 0, 0)      # 默认目标颜色 (R, G, B), 可通过 Q 热键动态修改。
                                   # 脚本启动时会自动从文件中加载上一次的颜色。
TOLERANCE = 10                    # 颜色容差，值越大，识别越宽松。

# 2. 性能参数
# 鼠标移动速度（秒），值越小移动越快
MOUSE_MOVE_DURATION = 0           # 普通移动速度
DRAG_MOVE_DURATION = 0            # 拖动时的移动速度（更快）
DEBUG_MOVE_DURATION = 0.1         # 调试模式下的移动速度

# 点击间隔（秒），值越小点击越快
CLICK_INTERVAL = 0                # 单次点击间隔
DRAG_POINT_INTERVAL = 0           # 拖动时点之间的间隔

# 扫描参数
SCAN_INTERVAL = 0.1               # 扫描间隔，值越小扫描越快但CPU占用更高

# 3. 绘图与优化
USE_SPACEBAR_DRAG = True          # 是否启用空格拖动优化。
PIXEL_ADJACENCY_THRESHOLD = 35    # 判断像素是否连续的距离阈值。
MIN_DRAG_POINTS = 4               # 启用拖动功能的最小连续点数

# 4. 窗口与区域配置
TARGET_WINDOW_TITLE = "Paint the world"
CROP_WINDOW_TOP = 145             # 从窗口顶部裁剪掉的高度，用于避开地址栏/书签栏。
EXCLUSION_ZONES = []              # 备用列表, 脚本会优先从 exclusion_zones.txt 加载。

# 5. 像素大小判定配置
SMALL_PIXEL_MAX_DIMENSION = 20    # 小像素点的最大尺寸
SMALL_PIXEL_MIN_AREA = 2          # 添加最小面积，避免噪声

# 6. 阈值调整步长配置
DIMENSION_ADJUST_STEP = 5         # 像素点阈值步长

# 7. 热键
HOTKEY_RUN_PAUSE = keyboard.KeyCode.from_char('w')      # W键：启停脚本
HOTKEY_PICK_COLOR = keyboard.KeyCode.from_char('q')     # Q键：取色
HOTKEY_TERMINATE = keyboard.KeyCode.from_char('g')      # G键：强制退出脚本
HOTKEY_DEBUG_SCAN = keyboard.KeyCode.from_char('x')     # X键：截取扫描截图
HOTKEY_MANUAL_ZONE = keyboard.KeyCode.from_char('h')    # H键：设置忽略区域（按两次确定区域）
HOTKEY_TOGGLE_DEBUG = keyboard.KeyCode.from_char('z')   # Z键：切换调试模式（只移动不点击）
HOTKEY_TOGGLE_DRAG = keyboard.KeyCode.from_char('t')    # T键：启停拖动功能
HOTKEY_DECREASE_THRESHOLD = keyboard.KeyCode.from_char('[')  # [键：减小像素阈值
HOTKEY_INCREASE_THRESHOLD = keyboard.KeyCode.from_char(']')  # ]键：增大像素阈值

# 8. 文件路径配置
ZONES_FILENAME = "exclusion_zones.txt"
DEBUG_FOLDER = "debug_scans"

# -------------------【全局状态变量】-------------------
script_active = False
terminate_script = False
targets_to_click = []
target_window_info = {}

# 扫描模式状态
DEBUG_MODE = False

# 手动区域选择状态
manual_zone_step = 0
manual_zone_p1 = (0, 0)

keyboard_controller = keyboard.Controller()

# -------------------【文件与区域管理】-------------------
def ensure_debug_folder():
    """确保调试文件夹存在"""
    if not os.path.exists(DEBUG_FOLDER):
        os.makedirs(DEBUG_FOLDER)
        logging.info(f"创建调试文件夹: {DEBUG_FOLDER}")

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

# -------------------【手动区域选择功能】-------------------
def handle_manual_zone():
    """处理手动区域选择：按两次H键确定区域"""
    global manual_zone_step, manual_zone_p1
    
    if manual_zone_step == 0:
        # 第一次按H：记录第一个点
        manual_zone_p1 = pyautogui.position()
        manual_zone_step = 1
        logging.info(f"区域选择(1/2): 已记录左上角坐标 {manual_zone_p1}。请移动鼠标到右下角再按 H。")
    else:
        # 第二次按H：记录第二个点并生成区域
        p2 = pyautogui.position()
        win_info = get_active_window_info()
        if not win_info:
            logging.warning("区域选择失败: 未找到目标窗口, 无法计算相对坐标。")
            manual_zone_step = 0
            return
        
        # 转换为相对于窗口内容区的坐标
        x1 = min(manual_zone_p1.x, p2.x) - win_info['left']
        y1 = min(manual_zone_p1.y, p2.y) - win_info['top']
        x2 = max(manual_zone_p1.x, p2.x) - win_info['left']
        y2 = max(manual_zone_p1.y, p2.y) - win_info['top']
        w, h = x2 - x1, y2 - y1
        
        zone_rect = (int(x1), int(y1), int(w), int(h))
        save_exclusion_data(zone_rect=zone_rect)
        manual_zone_step = 0
        logging.info(f"区域选择完成: 已保存忽略区域 {zone_rect}")

# -------------------【调试扫描功能】-------------------
def save_debug_scan(screenshot_cv, mask, valid_contours, all_contours, scan_info):
    """保存调试扫描结果"""
    ensure_debug_folder()
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # 创建调试图像
        debug_img = screenshot_cv.copy()
        
        # 1. 绘制所有找到的轮廓（红色）
        for contour in all_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2)  # 红色：所有轮廓
        
        # 2. 绘制有效的小像素轮廓（绿色）
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 3)  # 绿色：有效小像素
            # 标记中心点
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(debug_img, (center_x, center_y), 3, (0, 255, 0), -1)
        
        # 3. 绘制忽略区域（黄色）
        for zone in EXCLUSION_ZONES:
            zx, zy, zw, zh = zone
            cv2.rectangle(debug_img, (zx, zy), (zx+zw, zy+zh), (0, 255, 255), 2)  # 黄色：忽略区域
        
        # 4. 添加信息文本
        info_lines = [
            f"Target Color: {TARGET_COLOR_RGB}",
            f"Tolerance: {TOLERANCE}",
            f"Small Pixel Max Dim: {SMALL_PIXEL_MAX_DIMENSION}",
            f"All Contours: {len(all_contours)}",
            f"Valid Small Pixels: {len(valid_contours)}",
            f"Big Pixels Found: {scan_info.get('big_pixels', 0)}",
            f"Excluded by Zones: {scan_info.get('excluded_by_zones', 0)}",
            f"Timestamp: {timestamp}"
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 25
            cv2.putText(debug_img, line, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(debug_img, line, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # 保存调试图像
        debug_filename = f"debug_scan_{timestamp}.png"
        debug_filepath = os.path.join(DEBUG_FOLDER, debug_filename)
        cv2.imwrite(debug_filepath, debug_img)
        
        logging.info(f"调试扫描已保存: {debug_filename}")
        logging.info(f"扫描统计: 总轮廓{len(all_contours)}, 有效小像素{len(valid_contours)}, 大像素{scan_info.get('big_pixels', 0)}, 区域排除{scan_info.get('excluded_by_zones', 0)}")
        
        return debug_filepath
        
    except Exception as e:
        logging.error(f"保存调试扫描时发生错误: {e}")
        return None

# -------------------【窗口识别功能】-------------------
def get_active_window_info():
    """获取并返回当前活动窗口的信息, 用于坐标转换"""
    try:
        windows = gw.getWindowsWithTitle(TARGET_WINDOW_TITLE)
        if not windows: 
            logging.warning(f"未找到标题为 '{TARGET_WINDOW_TITLE}' 的窗口")
            return None
            
        window = windows[0]
        
        # 确保窗口坐标不为负
        win_x = max(0, window.left)
        win_y = max(0, window.top)
        
        # 计算裁剪后的窗口区域
        crop_top = CROP_WINDOW_TOP
        crop_height = max(0, window.height - crop_top)
        
        return {
            'left': win_x, 
            'top': win_y + crop_top, 
            'width': window.width, 
            'height': crop_height,
            'original_left': window.left,
            'original_top': window.top,
            'original_width': window.width,
            'original_height': window.height
        }
    except Exception as e:
        logging.error(f"获取窗口信息时发生错误: {e}")
        return None

# -------------------【核心绘画逻辑】-------------------
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
    
    # 优化：将点数少于MIN_DRAG_POINTS的组拆分成单点
    optimized_groups = []
    for group in groups:
        if len(group) >= MIN_DRAG_POINTS:
            optimized_groups.append(group)
        else:
            # 将小分组拆分成单点
            for point in group:
                optimized_groups.append([point])
    
    drag_groups = [g for g in optimized_groups if len(g) >= MIN_DRAG_POINTS]
    single_points = [g for g in optimized_groups if len(g) < MIN_DRAG_POINTS]
    
    logging.info(f"优化分析: {len(targets)}个目标点被分成了 {len(optimized_groups)} 组，其中 {len(drag_groups)} 组使用拖动，{len(single_points)} 个单点使用点击。")
    
    return optimized_groups

def scan_for_targets(debug_mode=False):
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
    
    # 筛选轮廓：排除忽略区域内的轮廓，并且只选择小像素
    valid_contours = []
    big_pixels_found = 0
    excluded_by_zones = 0
    too_small = 0
    area_stats = []
    size_stats = []
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        
        # 记录统计信息
        area_stats.append(area)
        size_stats.append((w, h))
        
        # 检查是否在忽略区域内
        if is_in_exclusion_zone(x, y, w, h):
            excluded_by_zones += 1
            continue
            
        # 检查基本尺寸条件
        if w <= 1 or h <= 1 or area < SMALL_PIXEL_MIN_AREA:
            too_small += 1
            continue
            
        # 检查是否是小像素
        if w < SMALL_PIXEL_MAX_DIMENSION and h < SMALL_PIXEL_MAX_DIMENSION:
            valid_contours.append(c)
        else:
            big_pixels_found += 1
    
    # 计算统计信息
    if area_stats:
        avg_area = sum(area_stats) / len(area_stats)
        max_area = max(area_stats)
        min_area = min(area_stats)
        logging.info(f"面积统计: 平均{avg_area:.1f}, 最小{min_area:.1f}, 最大{max_area:.1f}")
    
    # 保存调试信息
    scan_info = {
        'big_pixels': big_pixels_found,
        'excluded_by_zones': excluded_by_zones,
        'too_small': too_small,
        'area_stats': area_stats,
        'size_stats': size_stats
    }
    
    if debug_mode:
        debug_filepath = save_debug_scan(screenshot_cv, mask, valid_contours, contours, scan_info)
    
    # 显示详细统计信息（默认开启）
    logging.info(f"详细扫描统计: 总轮廓{len(contours)}, 有效小像素{len(valid_contours)}, 大像素{big_pixels_found}, 区域排除{excluded_by_zones}, 过小{too_small}")
    
    if not valid_contours:
        logging.info("扫描完成: 未找到满足条件的小像素块。")
        return True
        
    # 提取目标点
    temp_targets = []
    for c in valid_contours:
        x, y, w, h = cv2.boundingRect(c)
        temp_targets.append((x + w // 2, y + h // 2))

    if not temp_targets:
        logging.info("扫描完成: 未找到满足条件的标记。")
        return True
    
    temp_targets.sort(key=lambda p: (p[1], p[0]))
    targets_to_click = temp_targets
    logging.info(f"扫描完成: 找到 {len(targets_to_click)} 个小像素目标, 已生成任务列表。")
    return True

def is_in_exclusion_zone(x, y, w, h):
    """检查是否在忽略区域内"""
    rect_x2, rect_y2 = x + w, y + h
    for zx, zy, zw, zh in EXCLUSION_ZONES:
        zone_x2, zone_y2 = zx + zw, zy + zh
        if not (rect_x2 < zx or x > zone_x2 or rect_y2 < zy or y > zone_y2):
            return True
    return False

# -------------------【阈值调整功能】-------------------
def adjust_thresholds(increase=True):
    """调整像素判定阈值"""
    global SMALL_PIXEL_MAX_DIMENSION
    
    old_dimension = SMALL_PIXEL_MAX_DIMENSION
    
    if increase:
        # 增大阈值 - 识别更多小像素
        SMALL_PIXEL_MAX_DIMENSION += DIMENSION_ADJUST_STEP
        action = "增大"
    else:
        # 减小阈值 - 识别更少小像素
        SMALL_PIXEL_MAX_DIMENSION = max(5, SMALL_PIXEL_MAX_DIMENSION - DIMENSION_ADJUST_STEP)
        action = "减小"
    
    logging.info(f"阈值{action}: 尺寸 {old_dimension} -> {SMALL_PIXEL_MAX_DIMENSION}")

# -------------------【主循环与热键控制】-------------------
def pause_script(reason=""):
    """统一的暂停脚本函数"""
    global script_active, targets_to_click
    if script_active:
        script_active = False
        targets_to_click = [] # 暂停时清空任务列表
        logging.info(f"========== 脚本已 [暂停] (原因: {reason}) ==========")
        show_shortcuts()

def show_shortcuts():
    """显示快捷键说明"""
    logging.info("=== 快捷键说明 ===")
    logging.info("W: 启停脚本")
    logging.info("Q: 取色")
    logging.info("X: 扫描模式（仅截图分析）")
    logging.info("H: 设置忽略区域（按两次确定区域）")
    logging.info("Z: 切换调试模式（只移动不点击）")
    logging.info("T: 启停拖动功能")
    logging.info("[: 减小像素阈值（识别更少小像素）")
    logging.info("]: 增大像素阈值（识别更多小像素）")
    logging.info("G: 退出脚本")
    logging.info(f"当前阈值: 尺寸<{SMALL_PIXEL_MAX_DIMENSION}")

def on_press(key):
    """键盘按键处理"""
    global script_active, terminate_script, TARGET_COLOR_RGB
    global SMALL_PIXEL_MAX_DIMENSION, DEBUG_MODE, USE_SPACEBAR_DRAG
    global manual_zone_step, manual_zone_p1

    # 'G' 键: 退出脚本
    if key == HOTKEY_TERMINATE:
        logging.info("========== 检测到 'G' 键, 正在终止脚本... ==========")
        terminate_script = True
        return False

    # 'w' 键: 启动/暂停
    if key == HOTKEY_RUN_PAUSE:
        if script_active:
            pause_script("'w'键按下")
        else:
            script_active = True
            logging.info("========== 已按下 'w', 脚本已 [启动/继续] ==========")
        return

    # X 键: 截取扫描截图
    if key == HOTKEY_DEBUG_SCAN:
        logging.info("========== 执行扫描截图 ==========")
        scan_for_targets(debug_mode=True)
        return

    # H 键: 手动区域选择
    if key == HOTKEY_MANUAL_ZONE:
        handle_manual_zone()
        return

    # Z 键: 切换调试模式
    if key == HOTKEY_TOGGLE_DEBUG:
        DEBUG_MODE = not DEBUG_MODE
        status = "开启" if DEBUG_MODE else "关闭"
        logging.info(f"========== 调试模式已 {status} ==========")
        return

    # T 键: 切换拖动功能
    if key == HOTKEY_TOGGLE_DRAG:
        global USE_SPACEBAR_DRAG
        USE_SPACEBAR_DRAG = not USE_SPACEBAR_DRAG
        status = "开启" if USE_SPACEBAR_DRAG else "关闭"
        logging.info(f"========== 拖动功能已 {status} ==========")
        return

    # [ 键: 减小阈值
    if key == HOTKEY_DECREASE_THRESHOLD:
        adjust_thresholds(increase=False)
        return

    # ] 键: 增大阈值
    if key == HOTKEY_INCREASE_THRESHOLD:
        adjust_thresholds(increase=True)
        return

    # 在任何功能键按下时, 如果脚本在运行, 则先暂停
    if script_active:
        if key in [HOTKEY_PICK_COLOR, HOTKEY_MANUAL_ZONE]:
            pause_script(f"功能键 {key} 按下")
    
    # 'q' 键: 拾取颜色
    if key == HOTKEY_PICK_COLOR:
        pos = pyautogui.position()
        try:
            hover_color = pyautogui.pixel(pos.x, pos.y)
            # 直接使用悬停颜色
            TARGET_COLOR_RGB = hover_color
            
            save_exclusion_data(color_rgb=hover_color)
            logging.info(f"颜色选择: 鼠标位置 {pos}, 颜色 -> {hover_color}")
        except Exception as e:
            logging.error(f"取色失败: {e}")
        return

def start_keyboard_listener():
    """启动键盘监听器"""
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def main():
    """主函数"""
    ensure_debug_folder()
    load_exclusion_zones()
    
    # 测试窗口识别
    win_info = get_active_window_info()
    if win_info:
        logging.info(f"========== 窗口识别测试成功 ==========")
        logging.info(f"原始窗口: 位置({win_info['original_left']}, {win_info['original_top']}), 尺寸={win_info['original_width']}x{win_info['original_height']}")
        logging.info(f"工作区域: 位置({win_info['left']}, {win_info['top']}), 尺寸={win_info['width']}x{win_info['height']}")
        logging.info(f"=====================================")
    else:
        logging.error("========== 窗口识别测试失败 ==========")
        logging.error(f"请检查窗口标题 '{TARGET_WINDOW_TITLE}' 是否正确")
        logging.error(f"以及窗口是否处于打开状态")
        logging.error(f"=====================================")
    
    logging.info(f"赛博义体V999已启动")
    show_shortcuts()
    logging.info(f"调试模式: {'开启' if DEBUG_MODE else '关闭'}")
    logging.info(f"拖动功能: {'开启' if USE_SPACEBAR_DRAG else '关闭'} (最小连续点数: {MIN_DRAG_POINTS})")
    
    # 主循环将一直运行，直到 terminate_script 被设置为 True
    while not terminate_script:
        try:
            if not script_active:
                time.sleep(SCAN_INTERVAL)
                continue

            if not targets_to_click:
                if not scan_for_targets() or not targets_to_click:
                    pause_script("扫描结束或失败")
                    continue

            # --- 任务执行 ---
            task_groups = group_targets_for_drag(targets_to_click) if USE_SPACEBAR_DRAG else [[t] for t in targets_to_click]
            total_tasks = len(targets_to_click)
            tasks_done = 0
            
            for group in task_groups:
                if not script_active or terminate_script: break # 检查在处理组之间是否被暂停或终止
                
                # 移动到组的第一个点
                start_pos = group[0]
                final_x = target_window_info['left'] + start_pos[0]
                final_y = target_window_info['top'] + start_pos[1]
                
                # 根据是否调试模式选择移动速度
                move_duration = DEBUG_MOVE_DURATION if DEBUG_MODE else MOUSE_MOVE_DURATION
                pyautogui.moveTo(final_x, final_y, duration=move_duration)
                
                if len(group) == 1: # 单点, 直接点击
                    if not DEBUG_MODE: 
                        pyautogui.click()
                    logging.info(f"执行 {tasks_done+1}/{total_tasks} | 单点:({final_x},{final_y}) {'(调试)' if DEBUG_MODE else ''}")
                    # 单点点击后立即应用延迟
                    time.sleep(CLICK_INTERVAL)
                else: # 多点, 按住空格拖动
                    logging.info(f"执行 {tasks_done+1}-{tasks_done+len(group)}/{total_tasks} | 拖动画笔, 共 {len(group)} 点...")
                    
                    if not DEBUG_MODE: 
                        # 使用pynput的键盘控制器按下空格键
                        keyboard_controller.press(keyboard.Key.space)
                        time.sleep(0.05)
                    
                    try:
                        for i in range(1, len(group)):
                            if not script_active or terminate_script: break
                            point = group[i]
                            final_x = target_window_info['left'] + point[0]
                            final_y = target_window_info['top'] + point[1]
                            
                            # 根据是否调试模式选择移动速度
                            drag_duration = DEBUG_MOVE_DURATION if DEBUG_MODE else DRAG_MOVE_DURATION
                            pyautogui.moveTo(final_x, final_y, duration=drag_duration)
                            
                            # 拖动时每个点之间应用更短的延迟
                            time.sleep(DRAG_POINT_INTERVAL)
                    finally:
                        # 确保空格键被释放
                        if not DEBUG_MODE: 
                            time.sleep(0.05)  # 减少拖动结束前延迟
                            keyboard_controller.release(keyboard.Key.space)
                    
                    # 拖动完成后应用延迟
                    time.sleep(CLICK_INTERVAL)
                
                tasks_done += len(group)
            
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
