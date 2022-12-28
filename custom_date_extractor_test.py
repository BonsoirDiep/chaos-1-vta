from custom_date_extractor import date_extractor

def x(text):
    print(text, '>>', date_extractor.summary_date(text))

# x('ngày mai là quốc tế xyz')
# x('5 ngày nữa tôi thi do hôm qua tôi ngủ gật')
# x('do hôm qua tôi ngủ gật nên 5 ngày nữa tôi thi')
# x('5 ngày trước tôi có đi tham gia hội thảo khoa học trẻ')
x('5 tuần tới')

x('2 tháng vừa qua')

# x('ngày 30 tháng này')
# x('thứ năm tuần này')

# x('ngày 20 tháng vừa qua')
# x('ngày 20 năm tháng trước')

# x('tháng vừa qua')
# x('tháng này')
# x('tháng tới')

x('ngày 20 tháng vừa qua')
x('ngày 20 tháng này')
x('ngày 20 tháng tới')
x('ngày này tháng tới')

x('ngày này')

x('3 ngày trước tháng này')


x('ngày 20 tháng 2 năm tới')
x('ngày 20 tháng 2 năm 2023')

x('năm 2023')

x('năm tới')
x('năm vừa qua')

x('sau 4 tuần nữa')

x('năm cbc ngày hội sông nước hương giang ngày a ngày b ngày c')


# x('5 tháng trước')
# x('trong vòng 4 tháng tới')
# x('4 tháng tới')