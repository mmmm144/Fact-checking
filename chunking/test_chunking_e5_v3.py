from chunking_e5_v3 import clean_boilerplate_text, normalize_source_key


def test_bao_chinh_phu_related_suffix_is_removed() -> None:
    text = (
        "Nội dung bài viết hợp lệ.\nTác giả\n"
        "Tham khảo thêm Bài liên quan thứ nhất\n"
        "Tham khảo thêm Bài liên quan thứ hai"
    )

    assert clean_boilerplate_text(text, "BaoChinhPhu") == "Nội dung bài viết hợp lệ.\nTác giả"


def test_bao_chinh_phu_normal_prose_is_preserved() -> None:
    text = "Giáo viên cho biết học sinh có thể tham khảo thêm nhiều tài liệu khác."

    assert clean_boilerplate_text(text, "BaoChinhPhu") == text


def test_xem_them_inside_a_sentence_is_preserved() -> None:
    text = 'Khán giả nói: "Tôi muốn xem thêm nhiều video truyền cảm hứng".'

    assert clean_boilerplate_text(text, "VnExpress") == text


def test_standalone_xem_them_line_is_removed() -> None:
    text = "Nội dung chính.\nXem thêm: Bài viết khác"

    assert clean_boilerplate_text(text, "VnExpress") == "Nội dung chính."


def test_vnexpress_google_tutorial_and_related_links_are_removed() -> None:
    text = (
        "Nội dung chính.\nTác giả\n"
        ">> Bài viết liên quan\n"
        ">> Xem thêm nhiều video tại đây\n"
        "Hi\n"
        "Bước 1: Bấm vào nút ‘Thêm VnExpress trên Google’, hoặc mở liên kết.\n"
        "Bước 2: Chọn ô vuông bên phải."
    )

    assert clean_boilerplate_text(text, "VnExpress") == "Nội dung chính.\nTác giả"


def test_vnexpress_normal_numbered_steps_are_preserved() -> None:
    text = "Bước 1: Chuẩn bị hồ sơ.\nBước 2: Nộp hồ sơ tại cơ quan chức năng."

    assert clean_boilerplate_text(text, "VnExpress") == text


def test_source_normalization_accepts_accents_and_separators() -> None:
    assert normalize_source_key("Báo Chính phủ") == "baochinhphu"
