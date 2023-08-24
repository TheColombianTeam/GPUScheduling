from FaultInjector import golden
from FaultInjector import fault_list
from FaultInjector import injector
from FaultInjector import validator

from utils.args import args

#golden(120,120,120)
#fault_list()
#injector(args.workers)
validator(120,120)#this function needs output tensor dimentions inposed in golden for heat map bins 