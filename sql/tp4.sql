


SELECT *
FROM openquery([zabbix], '
select  col1 as dt,
		col2 
from public.history_uint 
where col_ = 6956180 
  and col1 >= extract(epoch from timestamp '@start') and col1 < extract(epoch from timestamp '@end')
order by col1 desc 
')











