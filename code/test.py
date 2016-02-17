import xlwt

dev = [1,6,7,8,9,0]
excel = xlwt.Workbook(encoding = "utf-8")
dev_sheet = excel.add_sheet("dev_sheet")
i = 0
for n in dev:
    dev_sheet.write(i, 0, n)
    i += 1
excel.save("dev_accuracy.xls")