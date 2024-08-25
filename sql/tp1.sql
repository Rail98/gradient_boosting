--declare @start datetime = '2024-06-05 00:00:00'
--declare @end datetime = '2024-06-06 00:00:00'

select distinct op.col1 as 'datetime', 
				op.col2 as 'oper_id'
from table1 op with (nolock)
join table2 con with (nolock) ON con.id_1 = op.id_1
join table3 pr with (nolock) ON pr.id_2 = con.id_2 and pr.col_ = 3
where op.col1 >= @start and op.col1 < @end
   and col_ in (2, 3, 17)
   and op.col_ not in ('2472E46C-03BC-4ED8-B8DF-074B22382AB4',
					   '9E3A3EA9-272A-4F4E-B790-53A896F644AA',
					   'FDC9467C-4EDC-46A3-9186-88D1B0BA7CC8',
					   '7D05259D-EEC8-4195-B51C-2F99DDBA92E6',
					   '0073E046-20E8-472F-BAC4-C8C57515B943',
					   'F0705F68-5F28-4FE4-AAB9-FF07C5E53750',
					   '53170294-579F-4BB8-ABBE-439823FB0EDB',
					   'BA83FF69-AAD0-40B1-A7C0-FCD4582AC417'
					 )
order by col_ desc