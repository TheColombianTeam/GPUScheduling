from FaultInjector import golden
from FaultInjector import fault_list
from FaultInjector import injector

golden(120,120,120)
fault_list()#by default faulty_SM = 0, faulty_cluster = 0
injector(5)