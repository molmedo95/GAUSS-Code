import sys
from py_cgr_lib.py_cgr_lib import Contact
from py_cgr_lib.py_cgr_lib import Route
from py_cgr_lib.py_cgr_lib import Bundle
from py_cgr_lib.py_cgr_lib import cp_load
from py_cgr_lib.py_cgr_lib import cgr_dijkstra

source = 1      # source node A
destination = 5 # destination node E
curr_time = 0   # Current time

contact_plan = cp_load('./contact_plans/cgr_tutorial.txt')

# dijkstra returns best route from contact plan
root_contact = Contact(source, source, 0, sys.maxsize, 100, 1.0, 0)  # root contact
root_contact.arrival_time = curr_time
route = cgr_dijkstra(root_contact, destination, contact_plan)

hops_list=[{'node':source,'time':curr_time}]
for hop in route.hops:
    hops_list.append({'node':int(hop.to),'time':hop.start})

print(hops_list)
