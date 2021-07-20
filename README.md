# ColorLabel

## Usage
1. 用繪圖板或滑鼠將原圖用**綠色**圈選欲標記的地方，並將標記好的圖放到同一資料夾下
2. 執行下方指令：
```
python label.py --input_path <path_to_input_dir> --output_path <path_to_output_dir> -c(--color)

Arguments:
<path_to_input_dir>：標記好的圖的資料夾路徑
<path_to_output_dir>：產生出來的mask的資料夾路徑
-c(--color): 透過圖片中的綠色來產生 mask
```

## Example
- Input
![](https://i.imgur.com/rKnFUW5.jpg)
- Output
![](https://i.imgur.com/MOdXCwh.jpg)
