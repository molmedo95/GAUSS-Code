# -*- coding: utf-8 -*-
# Librerias 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Miscelaneas
import sys
import numpy as np
import networkx as nx
import collections
import time

# Graph nets
from graph_nets import utils_np

# Py cgr
from py_cgr_lib.py_cgr_lib import Contact
from py_cgr_lib.py_cgr_lib import cgr_dijkstra
from py_cgr_lib.py_cgr_lib import cp_load_from_list



# Clases
class GraphPlotter(object):
  ''' La clase `GraphPlotter` concentra la funcionalidad para la visualización de grafos.
      Cuenta con métodos etiquetados con `@property` que permite acceder a las propiedades, computarlas si y solo si aún no han sido calculadas.
      La función es tomada tal como se provee en el ejemplo de la libraría GNN.
  '''

  def __init__(self, ax, graph, pos=None):
    self._ax = ax
    self._graph = graph
    self._pos = pos if pos!=None else nx.circular_layout(graph)
    self._base_draw_kwargs = dict(G=self._graph, pos=self._pos, ax=self._ax)
    self._solution_length = None
    self._nodes = None
    self._edges = None
    self._start_nodes = None
    self._end_nodes = None
    self._solution_nodes = None
    self._intermediate_solution_nodes = None
    self._solution_edges = None
    self._non_solution_nodes = None
    self._non_solution_edges = None
    self._ax.set_axis_off()

  @property
  def solution_length(self):
    if self._solution_length is None:
      self._solution_length = len(self._solution_edges)
    return self._solution_length

  @property
  def nodes(self):
    if self._nodes is None:
      self._nodes = self._graph.nodes()
    return self._nodes

  @property
  def edges(self):
    if self._edges is None:
      self._edges = self._graph.edges(keys=True)
    return self._edges

  @property
  def start_nodes(self):
    if self._start_nodes is None:
      self._start_nodes = [
          n for n in self.nodes if self._graph.nodes[n].get("frm", False)
      ]
    return self._start_nodes

  @property
  def end_nodes(self):
    if self._end_nodes is None:
      self._end_nodes = [
          n for n in self.nodes if self._graph.nodes[n].get("to", False)
      ]
    return self._end_nodes

  @property
  def solution_nodes(self):
    if self._solution_nodes is None:
      self._solution_nodes = [
          n for n in self.nodes if self._graph.nodes[n].get("solution", False)
      ]
    return self._solution_nodes

  @property
  def intermediate_solution_nodes(self):
    if self._intermediate_solution_nodes is None:
      self._intermediate_solution_nodes = [
          n for n in self.nodes
          if self._graph.nodes[n].get("solution", False) and
          not self._graph.nodes[n].get("frm", False) and
          not self._graph.nodes[n].get("to", False)
      ]
    return self._intermediate_solution_nodes

  @property
  def solution_edges(self):
    if self._solution_edges is None:
      self._solution_edges = [
          e for e in self.edges
          if self._graph.get_edge_data(e[0], e[1], e[2]).get("solution", False)
      ]
    return self._solution_edges

  @property
  def non_solution_nodes(self):
    if self._non_solution_nodes is None:
      self._non_solution_nodes = [
          n for n in self.nodes
          if not self._graph.nodes[n].get("solution", False)
      ]
    return self._non_solution_nodes

  @property
  def non_solution_edges(self):
    if self._non_solution_edges is None:
      self._non_solution_edges = [
          e for e in self.edges
          if not self._graph.get_edge_data(e[0], e[1], e[2]).get("solution", False)
      ]
    return self._non_solution_edges

  def _make_draw_kwargs(self, **kwargs):
    kwargs.update(self._base_draw_kwargs)
    return kwargs

  def _draw(self, draw_function, zorder=None, **kwargs):
    draw_kwargs = self._make_draw_kwargs(**kwargs)
    collection = draw_function(**draw_kwargs)
    if ((collection is not None) and (collection != [])) and (zorder is not None):
      try: # Compatibilidad
        collection.set_zorder(zorder)
      except AttributeError:
        collection[0].set_zorder(zorder)
    return collection

  # Dibujar nodos
  def draw_nodes(self, **kwargs):
    """Useful kwargs: nodelist, node_size, node_color, linewidths."""
    if ("node_color" in kwargs and
        isinstance(kwargs["node_color"], collections.Sequence) and
        len(kwargs["node_color"]) in {3, 4} and
        not isinstance(kwargs["node_color"][0],
                       (collections.Sequence, np.ndarray))):
      num_nodes = len(kwargs.get("nodelist", self.nodes))
      kwargs["node_color"] = np.tile(
          np.array(kwargs["node_color"])[None], [num_nodes, 1])
      
    return self._draw(nx.draw_networkx_nodes, **kwargs)

  # Dibujar arcos
  def draw_edges(self, **kwargs):
    """Useful kwargs: edgelist, width."""
    return self._draw(nx.draw_networkx_edges, **kwargs)

  def draw_graph(self,
                 node_size=200,
                 node_color=(0.4, 0.8, 0.4),
                 node_linewidth=1.0,
                 edge_width=1.0):
    # Plot nodes.
    self.draw_nodes(
        nodelist=self.nodes,
        node_size=node_size,
        node_color=node_color,
        linewidths=node_linewidth,
        zorder=20)
    # Plot edges.
    self.draw_edges(edgelist=self.edges, width=edge_width, zorder=10)

  def draw_graph_with_solution(self,
                               node_size=200,
                               node_color=(0.4, 0.8, 0.4),
                               node_linewidth=1.0,
                               edge_width=1.0,
                               start_color="w",
                               end_color="k",
                               solution_node_linewidth=3.0,
                               solution_edge_width=3.0):
    node_border_color = (0.0, 0.0, 0.0, 1.0)
    node_collections = {}
    # Plot start nodes.
    node_collections["start nodes"] = self.draw_nodes(
        nodelist=self.start_nodes,
        node_size=node_size,
        node_color=start_color,
        linewidths=solution_node_linewidth,
        edgecolors=node_border_color,
        zorder=100)
    # Plot end nodes.
    node_collections["end nodes"] = self.draw_nodes(
        nodelist=self.end_nodes,
        node_size=node_size,
        node_color=end_color,
        linewidths=solution_node_linewidth,
        edgecolors=node_border_color,
        zorder=90)
    # Plot intermediate solution nodes.
    if isinstance(node_color, dict):
      c = [node_color[n] for n in self.intermediate_solution_nodes]
    else:
      c = node_color
    node_collections["intermediate solution nodes"] = self.draw_nodes(
        nodelist=self.intermediate_solution_nodes,
        node_size=node_size,
        node_color=c,
        linewidths=solution_node_linewidth,
        edgecolors=node_border_color,
        zorder=80)
    # Plot solution edges.
    node_collections["solution edges"] = self.draw_edges(
        edgelist=self.solution_edges, width=solution_edge_width, edge_color="b", zorder=70)
    # Plot non-solution nodes.
    if isinstance(node_color, dict):
      c = [node_color[n] for n in self.non_solution_nodes]
    else:
      c = node_color
    node_collections["non-solution nodes"] = self.draw_nodes(
        nodelist=self.non_solution_nodes,
        node_size=node_size,
        node_color=c,
        linewidths=node_linewidth,
        edgecolors=node_border_color,
        zorder=20)
    # Plot non-solution edges.
    node_collections["non-solution edges"] = self.draw_edges(
        edgelist=self.non_solution_edges, width=edge_width, zorder=10)
    # Set title as solution length.
    self._ax.set_title("Saltos camino mas corto: {}".format(self.solution_length))
    return node_collections



# Funciones
def set_diff(seq0, seq1):
  ''' Devuelve los valores presentes en seq0 pero no en seq1.
  
  Parameters:
    seq0 <iterable>
    seq1 <iterable>
  
  Returns:
    <list>: Lista con los valores presentes en seq0 pero no en seq1
  '''

  return list(set(seq0) - set(seq1))


def to_one_hot(indice, max_value, axis=-1):
  ''' Returns a vector of length "max_value", with a "1" at the position indicated by "index".
  
    parameters:
      index <int>: input value to encode in one_hot
      max_value <int>: maximum number of values to encode (2 for true or false)
      axis <int, optional, default=-1>
  
    returns:
      one_hot <numpy.ndarray>: Encoding one hot
  '''

  one_hot = np.eye(max_value)[indice]
  if axis not in (-1, one_hot.ndim):
    one_hot = np.moveaxis(one_hot, -1, axis)
  return one_hot


def load_contactplan(path_ContactPlan):
  ''' Carga el plan de contactos en np.array desde un archivo
  
  Parameters:
    path_ContactPlan <str>: Ruta al archivo del plan de contactos
  
  Returns:
    <numpy.ndarray>: np.array con el listado de contactos, formato: "[start, end, frm, to, rate, owlt]"
  
  Notes:
    Se debe procurar que la lista "contact" este en el mismo orden que la lista "range"
  '''

  contact_plan = []
  i_aux = 0
  with open(path_ContactPlan, 'r') as file:
    for contact in file.readlines():
      if contact[0] == '#':
        continue
      if contact.startswith('a contact'):
        fields  = contact.split(' ')[2:]  # ignore "a contact"
        start, end, frm, to, rate = map(float, fields)
        owlt = 0.1
        contact_plan.append([int(start), int(end), int(frm-1), int(to-1), int(rate), owlt])
      if contact.startswith('a range'):
        fields  = contact.split(' ')[2:]  # ignore "a range"
        start, end, frm, to, owlt = map(float, fields)
        contact_plan[i_aux][5] = owlt
        i_aux += 1
  return np.array(contact_plan)


def generate_rand_param(rand, contactPlan, t_window, t_inicial=None, node_A=None, node_B=None):
  ''' Genera de forma aleatoria los parametros t_inicial, nodo origen y nodo destino
  
  Parameters:
    rand <rand obj>: Maquina generadora de numeros aleatorios
    contactPlan <numpy.ndarray>: Listado de contactos
    t_window <int>: Longitud en segundos de la ventana a considerar
    t_inicial <int, optional, default=None>: Tiempo inicial, si es igual a None se toma un tiempo aleatorio
    node_A <int, optional, default=None>: Nodo de origen, en caso de ser None se selecciona aleatoriamente
    node_B <int, optional, default=None>: Nodo de destino, en caso de ser None se selecciona aleatoriamente
  
  Returns:
    t_inicial <float>: T inicial de la ventana temporal
    t_final <float>: T final de la ventana temporal
    node_A <int>: Nodo origen
    node_B <int>: Nodo destino
  '''

  # Extraigo info necesaria desde el contactPlan
  t_max = contactPlan[:,1].max()
  listOfNodes = np.unique(contactPlan[:,2:4]).astype(int)
  
  # Determino los limites de la ventana temporal
  t_inicial_max = t_max - t_window
  if (t_inicial==None):
    t_inicial = rand.randint(t_inicial_max)
  else:
    if (t_inicial > t_inicial_max):
      raise ValueError("t_inicial inconsistent with t_window")
  t_final = t_inicial + t_window

  # Selecciono los nodos de inicio y de fin
  if node_A==None:
    node_A = rand.choice(listOfNodes[listOfNodes!=node_B])
  else:
    if not node_A in listOfNodes:
      raise ValueError("node A, is not found in the list of valid nodes")
  
  if node_B==None:
    node_B = rand.choice(listOfNodes[listOfNodes!=node_A])
  else:
    if not node_B in listOfNodes:
      raise ValueError("node B, is not found in the list of valid nodes")
  
  if node_A==node_B:
    raise ValueError("The destination node cannot be the same as the source node")

  return t_inicial, t_final, node_A, node_B


def generate_graph(contactPlan, t_inicial, t_final, node_A, node_B):
  ''' Genera un grafo apartir de un periodo especifico del contact plan [t_inicial, t_final].
  
  Parameters:
    contactPlan <numpy.ndarray>: Listado de contactos
    t_inicial <int>: Tiempo inicial
    t_final <int>: Tiempo final
    node_A <int>: Nodo de origen
    node_B <int>: Nodo de destino
  
  Returns:
    graph_from_cp <networkx.Multidigraph>: Grafo generado apartir del sub contact plan
    sub_contact_plan <numpy.ndarray>: Sub contact plan utilzado
  '''
  
  # Creo un sub contact plan
  subContactPlan = contactPlan[(contactPlan[:,1] > t_inicial) & (contactPlan[:,0] < t_final)]
  del(contactPlan)

  # Normalizo los tiempos, t=0 equivale a t_inicial
  subContactPlan[:,0]-=t_inicial
  subContactPlan[:,1]-=t_inicial
  subContactPlan[subContactPlan[:,0]<0, 0] = 0
  subContactPlan[subContactPlan[:,1]>(t_final-t_inicial), 1] = (t_final-t_inicial)

  # Cargo los nodos al grafo y asigno sus atributos "frm" y "to"
  graph = nx.MultiDiGraph()
  graph.add_nodes_from(np.unique(subContactPlan[:,2:4]).astype(int), frm=False, to=False)
  graph.add_node(node_A, frm=True)
  graph.add_node(node_B, to=True)

  # Cargo todos los edge con sus atributos
  for i in subContactPlan:
    graph.add_edge(int(i[2]), int(i[3]),key=float(i[0]), start=int(i[0]), end=float(i[1]), rate=int(i[4]), owlt=float(i[5]))

  # Cargo atributos globales
  graph.graph['t_inicial'] = float(t_inicial)

  return graph, subContactPlan


def add_cgr_path(graph, subContactPlan):
  ''' Calcula la mejor ruta entre un nodo A y otro nodo B usando el algoritmo de CGR
  
  Parameters:
    graph <networkx.MultiDiGraph>: Grafo de entrada con los nodos A y B etiquetados como frm y to
    sub_contact_plan <numpy.ndarray>: Sub contact plan utilizado para generar el grafo
  
  Returns:
    labeled_graph <networkx.MultiDiGraph>: Grafo de salida, identico al grafo de entrada pero con la ruta etiquetada
    ElapsedTimeCGR <float>: Tiempo requerido para el calculo de la ruta con el algortmo CGR
  '''

  # Extraigo del grafo los nodos frm y to
  for i in graph.nodes.data():
    if i[1]['frm']:
      node_A = i[0]
    if i[1]['to']:
      node_B = i[0]
  
  # Computo el camino mas corto.
  ElapsedTimeCGR = time.time()
  root_contact = Contact(node_A, node_A, 0, sys.maxsize, 100, 1.0, 0)  # root contact
  root_contact.arrival_time = 0
  subContactPlan_for_cgr = cp_load_from_list(subContactPlan.tolist())
  route = cgr_dijkstra(root_contact, node_B, subContactPlan_for_cgr)
  ElapsedTimeCGR = time.time() - ElapsedTimeCGR

  if route == None:
    return None, 0
  
  path_nodes = []
  path_edges = []
  for hop in route.hops:
    path_nodes.append(int(hop.frm))
    path_edges.append((int(hop.frm), int(hop.to), hop.start))
  path_nodes.append(int(hop.to))

  # Creamos un nuevo grafo para almacenar el camino encontrado.
  target_graph = graph.to_directed()

  # Marco el atributo "solution" de cada nodo y edge.
  target_graph.add_nodes_from(set_diff(target_graph.nodes(), path_nodes), solution=False)
  target_graph.add_nodes_from(path_nodes, solution=True)
  target_graph.add_edges_from(set_diff(target_graph.edges, path_edges), solution=False)
  target_graph.add_edges_from(path_edges, solution=True)

  # Guardo como atributo global del grafo path_nodes y path_edges
  target_graph.graph['path_nodes'] = path_nodes
  target_graph.graph['path_edges'] = path_edges

  return target_graph, ElapsedTimeCGR


def add_features_input(graph):
  ''' Apartir de "graph" genera el grafo de entrada (input_graph) el cual incluye el atributo "features" el cual codifica en 
  forma de vector la informacion de entrada a la red neuronal.
  
  Parameters:
    graph <networkx.MultiDiGraph>: Grafo de entrada
    
  Returns:
    input_graph <networkx.MultiDiGraph>: Grafo a utilizar como entrada de la red neuronal
  
  Notes:
    El vector de entrada esta formado por los atributos ("frm", "to") para los nodos, y ("start", "end") para los edges.
  '''

  def create_feature(attr, fields):
    return np.hstack([np.array(attr[field], dtype=float) for field in fields])

  input_node_fields = ("frm", "to")
  input_edge_fields = ("start", "end")
  
  input_graph = graph.copy()

  for node_index, node_feature in graph.nodes(data=True):
    input_graph.add_node(node_index, features=create_feature(node_feature, input_node_fields))

  for sender, receiver, key, features in graph.edges(keys=True, data=True):
    input_graph.add_edge(sender, receiver, key=key, features=create_feature(features, input_edge_fields))

  input_graph.graph["features"] = np.array([0.0])

  return input_graph


def add_features_target(graph):
  ''' Apartir de "graph" genera el grafo target (target_graph) el cual incluye el atributo "features" el cual codifica en 
  forma de vector el ground truth utilizado para la evaluacion
  
  Parameters:
    graph <networkx.MultiDiGraph>: Grafo de entrada
    
  Returns:
    target_graph <networkx.MultiDiGraph>: Grafo a utilizar como ground truth
  
  Notes:
    El ground truth esta codificado en one hot.
  '''

  target_node_fields = ("solution")
  target_edge_fields = ("solution")

  target_graph = graph.copy()

  solution_length = 0
  for node_index, node_feature in graph.nodes(data=True):
    target_node = to_one_hot(int(node_feature[target_node_fields]), 2)
    target_graph.add_node(node_index, features=target_node)
    solution_length += int(node_feature["solution"])
  solution_length /= graph.number_of_nodes()

  for sender, receiver, key, features in graph.edges(keys=True, data=True):
    target_edge = to_one_hot(int(features[target_edge_fields]), 2)
    target_graph.add_edge(sender, receiver, key=key, features=target_edge)

  target_graph.graph["features"] = np.array([solution_length], dtype=float)

  return target_graph


def compute_accuracy(output, target):
  ''' Computa distintas metricas (Precision, Recall, F1), para los edge y los nodos.  
  (Se promedian los valores obtenidos para cada uno de los grafos de entrada)
  
  Parameters:
    target <graph_nets.GraphsTuple>: ground truth
    output <graph_nets.GraphsTuple>: output de la red neuronal
    
  Returns:
    output <dict>: Diccionarios con las metricas: 'accuracy_graph', 'precision_nodes', 'recall_nodes', 'f1_nodes', 'precision_edges', 'recall_edges', 'f1_edges'.
    accuracyVsHops <dict>: Diccionario con el accuracy obtenido para cantidad de hops
  '''
  
  odds = utils_np.graphs_tuple_to_data_dicts(output)
  tdds = utils_np.graphs_tuple_to_data_dicts(target)

  ss=[]
  accuracy  = 0
  precision_nodes = []
  recall_nodes    = []
  f1_nodes        = []
  precision_edges = []
  recall_edges    = []
  f1_edges        = []
  accuracyVsHops  = {}

  for od, td in zip(odds, tdds):
    xn = np.argmax(od["nodes"], axis=-1)
    yn = np.argmax(td["nodes"], axis=-1)
    xe = np.argmax(od["edges"], axis=-1)
    ye = np.argmax(td["edges"], axis=-1)

    # nodes
    true_positive = float(((xn==1)&(yn==1)).sum())
    false_positive= float(((xn==1)&(yn==0)).sum())
    false_negative= float(((xn==0)&(yn==1)).sum())

    d_aux=(true_positive+false_positive)
    if d_aux!=0:
      p = true_positive/d_aux
    else:
      p = 0.0001

    d_aux=(true_positive+false_negative)
    if d_aux!=0:
      r = true_positive/d_aux
    else:
      r = 0.0001

    d_aux=(p+r)
    if d_aux!=0:
      f = (2*(p*r))/(p+r)
    else:
      f = 0.0001
    
    precision_nodes.append(p)
    recall_nodes.append(r)
    f1_nodes.append(f)


    # edges
    true_positive = float(((xe==1)&(ye==1)).sum())
    false_positive= float(((xe==1)&(ye==0)).sum())
    false_negative= float(((xe==0)&(ye==1)).sum())

    d_aux=(true_positive+false_positive)
    if d_aux!=0:
      p = true_positive/d_aux
    else:
      p = 0.0001

    d_aux=(true_positive+false_negative)
    if d_aux!=0:
      r = true_positive/d_aux
    else:
      r = 0.0001

    d_aux=(p+r)
    if d_aux!=0:
      f = (2*(p*r))/(p+r)
    else:
      f = 0.0001

    precision_edges.append(p)
    recall_edges.append(r)
    f1_edges.append(f)


    # graph
    c = []
    c.append(xn == yn)
    c.append(xe == ye)
    c = np.concatenate(c, axis=0)
    s = np.all(c)
    ss.append(s)


    # Accuracy vs Hops
    Hops = int(np.sum(ye))
    if Hops in accuracyVsHops:
      accuracyVsHops[Hops].append(s)
    else:
      accuracyVsHops[Hops] = [s]

  accuracy        = np.mean(np.stack(ss))
  precision_nodes = np.mean(precision_nodes)
  recall_nodes    = np.mean(recall_nodes)
  f1_nodes        = np.mean(f1_nodes)
  precision_edges = np.mean(precision_edges)
  recall_edges    = np.mean(recall_edges)
  f1_edges        = np.mean(f1_edges)

  for i in accuracyVsHops:
    accuracyVsHops[i] = np.mean(accuracyVsHops[i])
  
  output = {'accuracy_graph':accuracy, 'precision_nodes':precision_nodes, 'recall_nodes':recall_nodes, 'f1_nodes':f1_nodes, 
            'precision_edges':precision_edges, 'recall_edges':recall_edges, 'f1_edges':f1_edges} 

  accuracyVsHops = collections.OrderedDict(sorted(accuracyVsHops.items()))
  
  return output, accuracyVsHops

