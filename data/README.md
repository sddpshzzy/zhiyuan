# Local input contract

Place the following files in one local directory:

- `实验数据.xlsx`: sheet `化验表` supplies the original TFe assay intervals
- `定位.xlsx`
- `侧斜.xlsx`
- `岩性.xlsx`
- `整合钻孔品位模型表_审计用.xlsx` (legacy audit only)
- `Processed_Point_Data_审计用.xlsx` (legacy audit only)

The first four files are used to construct the corrected modeling table. The assay workbook provides the assay intervals; the three standalone workbooks provide collar locations, downhole surveys, and lithological intervals. The two legacy files are inspected solely to document the former mean-filling problem and are never used for model fitting.

Do not commit raw or interval-level processed mine data. The root `.gitignore` excludes common local data outputs.
