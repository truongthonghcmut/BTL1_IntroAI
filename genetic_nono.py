import random as rd
import numpy as np

# Lấy thông tin về nonogram
def getInfomation(nonogram):
    return nonogram.get("rows"), nonogram.get("cols"), \
    [row_clue for row in nonogram.get("rows") for row_clue in row], \
    [col_clue for col in nonogram.get("cols") for col_clue in col]

# Khởi tạo kiểu gen cho mỗi cột của nonogram dưới dạng danh sách các kiểu gen
def defineGenotypeForEachIndividual(cols):
    genotype = []
    for col in cols:
        genotype.append({
            "col": col,
            "genome": [rd.randrange(100) for _ in range(len(col))]
        })
    return genotype

# Giải mã từ kiểu gen thành nonogram 2D
def decodeGenotype(rows, genotype):
    decoded_2D_table = []
    for element in genotype:
        col = element.get("col")
        genome = element.get("genome")
        decoded_col = [[] for col_index in range(len(rows))]
        # Biến quản lý vị trí của gen
        location = {
            "start_index": 0, # Vị trí bắt đầu của gen
            "allowed_index": None, # Số vị trí có thể đặt gen
            "placed": None # Vị trí đặt gen được chọn
        }
        for index, clue in enumerate(col):
            location["allowed_index"] = 0
            # Tìm giới hạn xa nhất có thể đặt
            right_bound_index = (len(decoded_col) - 1) - (sum(col[index + 1:]) + len(col[index + 1:]))
            # Tìm giới hạn gần nhất có thể đặt
            left_bound_index = int(next((index + 2 for index, element in reversed(list(enumerate(decoded_col))) if element), '0'))
            # Đếm số lượng vị trí hợp lệ có thể đặt gen
            for index_place, place in enumerate(decoded_col[left_bound_index: right_bound_index + 1]):
                if index_place + left_bound_index + clue <= right_bound_index + 1: location["allowed_index"] += 1
            # Chọn vị trí đặt gen ngẫu nhiên (sử dụng modulo)
            location["placed"] = location.get("start_index") + genome[index] % location.get("allowed_index")
            # Đặt gen vào vị trí đã chọn 
            for index_place, place in enumerate(decoded_col[location.get("placed"): location.get("placed") + clue]): place.append('x')
            # Cập nhật vị trí bắt đầu cho nhóm tiếp theo, đảm bảo có ít nhất 1 ô trống
            location["start_index"] = location.get("placed") + clue + 1
        decoded_2D_table.append(decoded_col)
    return decoded_2D_table

# Chuyển đổi nonogram sau khi giải mã thành danh sách các gợi ý (solution) cho từng cột
def getSolution(cols, decoded_genotype):
    solution = [[] for col_index in range(len(cols))]
    column = 0
    for col in decoded_genotype:
        cnt = 0
        for index, element in enumerate(col):
            # Xử lý dòng đầu tiên: nếu trống thì tăng đếm lên 1, còn nếu có ô 'x' thì thêm số đếm này vào cột và reset lại số đếm
            if index == 0:
                if not element: cnt = cnt + 1
                else:
                    solution[column].append(cnt)
                    cnt = 0
            # Xử lý các dòng tiếp theo: nếu trống thì tăng đếm lên 1, với các ô đen: nếu đã có ô đen trước đó thì lưu count - 1, ngược lại lưu count
            else:
                if not element: cnt = cnt + 1
                elif cnt == 0: pass
                elif next((True for element in col[:index] if element), False):
                    solution[column].append(cnt - 1)
                    cnt = 0
                else:
                    solution[column].append(cnt)
                    cnt = 0
        column = column + 1
    return solution

# Lấy thông tin về nonogram từ cá thể
def getNonogramInfomationFromIndividual(rows, col_blocks, individual):
    decoded_individual = []
    col_index = 0
    for col in individual:
        decoded_col = [[] for i in range(len(rows))]
        for index, block in enumerate(col):
            # Nếu là nhóm ô đầu tiên, đặt dấu 'x' từ block đến block + col_blocks[col_index]
            if index == 0:
                for place in range(block, block + col_blocks[col_index]): decoded_col[place].append('x')
            # Nếu là nhóm ô tiếp theo, tìm vị trí trống kế tiếp (last_place) để bắt đầu đặt 'x'
            else:
                last_place = next((index + 2 for index, element in reversed(list(enumerate(decoded_col))) if element))
                for place in range(last_place + block, last_place + block + col_blocks[col_index]): decoded_col[place].append('x')
            col_index = col_index + 1
        decoded_individual.append(decoded_col)
    # Đổi qua từ danh sách cột sang danh sách hàng (ma trận chuyển vị)
    arr = np.array(decoded_individual, dtype=object)
    transpose_matrix = arr.T
    return transpose_matrix.tolist()

# Lấy thông tin về hàng từ nonogram
def getRowsFromNonogram(nonogram):
    to_strings = []
    for row in nonogram:
        row_string = ""
        for cell in row:
            if cell: row_string += 'x'
            else: row_string += ' '
        to_strings.append(row_string)
    result = []
    for s in to_strings:
        row_counts = []
        count = 0
        for char in s:
            if char == 'x':count = count + 1
            else:
                if count > 0:
                    row_counts.append(count)
                    count = 0
        if count > 0:row_counts.append(count)
        result.append(row_counts)
    return result