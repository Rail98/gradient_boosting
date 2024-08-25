SELECT t1.[col1] as 'Номер обращения'
      ,t2.[col2] as 'Техпроцесс'
      ,t1.[col3] as 'Сервис'
      ,t1.[col4] as 'Фактдата начала'
      ,t1.[col5] as 'Фактдата окончания'
FROM [DW-IT4IT-PROD].[DataWarehouse].[dbo].[table1] t1
join [ITPerfomanceMetrics].[dbo].[table2] t2 on t1.[col3] = t2.[col3]
where t1.[col4] >= '2024-01-01'
  and t1.[updated] >= dateadd(HH, -24, GETDATE())





