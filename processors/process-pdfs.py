import tabula

# df = tabula.read_pdf('./data/pdfs/Salary-List-September-2019.pdf', pages='all')
tabula.convert_into('./data/pdfs/Salary-List-September-2019.pdf', './data/2019-salaries.csv', output_format="csv", pages='all')