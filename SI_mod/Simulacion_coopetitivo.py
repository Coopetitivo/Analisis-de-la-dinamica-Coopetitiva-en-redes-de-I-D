import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import scipy.stats as ss
import heapq

def corrector(matriz):

    Matriz = copy.deepcopy(matriz)
    i = 0
    for rows in matriz.index:
        
        a = Matriz.loc[rows, :].to_numpy()[i:len(matriz.index)]
        Matriz.iloc[[v for v in range(i,len(matriz.index))], i] = a
        Matriz.loc[rows, rows] = 0
        i += 1
        
    return(Matriz)

def mat_power(n, matriz):

    m_p = copy.deepcopy(matriz)

    for i in range(n):

        m_p = m_p.dot(matriz)

    return(m_p)

def make_prob(Matriz):

    matriz = copy.deepcopy(Matriz)

    for rows in Matriz.index:

        matriz.loc[rows, :] = Matriz.loc[rows,:]/(Matriz.loc[rows,:].sum())
    
    return(matriz)

def make_jaccard(Matriz, coeficientes):

    matriz = copy.deepcopy(Matriz)

    for rows in Matriz.index:

        a = coeficientes.transpose()*Matriz.loc[rows, :]
        #print(a)
        #a.index = rows

        a = a.squeeze()

        matriz.loc[rows, :] = a.transpose()

    return (matriz)

def graph(conexiones,clase,coeficientes=[]):
    
    fig, ax = plt.subplots()
    
    if coeficientes == []:
        
        pesot = conexiones.sum()
        
    else: 
        
        pesot = conexiones.sum()*coeficientes

    nexos = conexiones.to_dict()

    pos = {}
    
    tipo = list(np.unique(clase))
    
    clase = clase
    clases = np.unique(clase.loc[[i[0] for i in nexos],:])
    
    entidades = {}
    
    for i in range(len(nexos)):
        
        entidades.update({f"{i+1}":conexiones.index[i][0]})
    
    
    color = {}
    
    colortipo = []
    
    colory = ["tomato", "mediumseagreen", "deepskyblue", "coral", "teal"]

    for i in nexos:
        
        a = colory[tipo.index(clase.loc[i,:].to_numpy()[0])]
        color.update({f"{i[0]}":
                      a})
    for i in clases:
        
        plt.plot([1],[1], color = colory[tipo.index(i)], linewidth=1,
                 label = i)

    c = 0
    
    for i in nexos:
        
        u = np.random.uniform(0,2*(c+1),1)
        u1 = np.random.uniform(0, 1,1)
        
        pos.update({f"{i[0]}" : [u, u1]})

        c +=1
    
    edge = {}

    for i in nexos:

        edge.update({f"{i[0]}" : []})

        for j in nexos[i]:

            if nexos[i][j] != 0:

                edge[f"{i[0]}"].append(f"{j[0]}")

    c = 0
    #k = 10*np.pi/np.max(pesot)

    for i in edge:

 #       color = plt.cm.RdYlBu(np.sin(k*int(pesot[c])))

        for j in edge[i]:
            
            valor_medio=[(pos[i][0]+pos[j][0])/2, (pos[i][1]+pos[j][1])/2]

            plt.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color=color[f"{i}"], linewidth=0.2, zorder=-1)
            
            #plt.text(valor_medio[0], valor_medio[1], str(conexiones.loc[i,j].to_numpy()[0][0]), fontsize=3)
            
        c += 1

    c = 1

    for i in nexos:

      #  color = plt.cm.RdYlBu(np.sin(k*int(pesot[c-1])))

        size = pesot[c-1]*(len(conexiones)/43)

        plt.scatter(pos[f"{i[0]}"][0], pos[f"{i[0]}"][1], s=size, color=color[f"{i[0]}"], zorder=1)

        plt.text(pos[f"{i[0]}"][0], pos[f"{i[0]}"][1], entidades[f"{c}"], fontsize=4)

        c += 1
    
    for i in tipo:
        
        colortipo.append(plt.cm.RdYlBu(tipo.index(i)/10))
        
        #plt.scatter( 10, -2, color = colortipo[tipo.index(i)], label = i)
    
    #plt.legend(tipo, loc="lower right")
    
    plt.legend(prop={"size":5})
    
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    return([pos, edge, color])


def markov(init, matriz, t, condicion=0 ,estados=[]):

    if len(estados) == 0:

        estados = np.arange(len(init))

    if condicion == 0:

        condicion = estados[-1]

    estados = estados
    simlist = np.zeros(t+1)
    simlist = [list(np.random.choice(estados, 1, p=init))]

    for i in range(1, t+1):#https://people.carleton.edu/~rdobrow/StochasticBook/Rscripts/utilities.R

        if condicion in simlist:

            break

        simlist.append(list(np.random.choice(estados, 1, p=matriz.loc[simlist[i-1], :].to_numpy()[0])))

    return(simlist)

def DTMC_SIS(beta, gamma, poblacion, c_tiempo, duracion): #Markov_Chain_epidemic_models_and_Parameter_Estimation

    #Los infectados puede ser un vector donde se almacene un 
    #vector de valores unicos
    #Las reeinfecciones no se cuentan
    #Entonces podemos usar el mismo modelo
    #donde I sea el vector de infectados
    #S el vector de entidades susceptibles
    #El ultimo contagia, podrías cambiar el beta y gamma posiblemente
    dt = int(duracion/c_tiempo)
    M_T = np.matrix(np.zeros((dt,dt)))
    M_T[0,0]=1
    Tc = []
    
    tc = c_tiempo
    
    Tc.append(c_tiempo)
    
    I=np.zeros(dt+1)
    S=np.zeros(dt+1)
    I[0]=1
    S[0]=poblacion-I[0]
    P = []
    
    #Añadir un ciclo for aqui para la lista de simulacion

    for t in range(dt):

        p1=(beta*S[t]*I[t]/poblacion)*tc
        P.append(p1)
        
        p2=(gamma*I[t]*tc)
        
        q = 1-(p1)

        if len(M_T)-1 >= t+1:

            M_T[t+1,t+1]=q
            #M_T[t,t+1]=p2

            if len(M_T)-1 >= t+2:
                
                M_T[t+2,t+1]=p1

        #print(M_T)

        u = np.random.uniform(0, 1, 1)
        
        u = u[0]

        if 0 < u and u <= p1:

            I[t+1] = I[t]+1
            
            S[t+1] = S[t]-1
            

        #elif p1 < u and u <= (p1+p2):

            #I[t+1] = I[t]-1
            
            #S[t+1] = S[t]+1          

        else:

            S[t+1] = S[t]
            I[t+1] = I[t]
            
        Tc.append(Tc[t]+c_tiempo)
    
    return ([I,S,pd.DataFrame(M_T).transpose(),Tc])

def DTMC_SI(beta, poblacion, c_tiempo, duracion, infectado): #Markov_Chain_epidemic_models_and_Parameter_Estimation

    #Los infectados puede ser un vector donde se almacene un 
    #vector de valores unicos
    #Las reeinfecciones no se cuentan
    #Entonces podemos usar el mismo modelo
    #donde I sea el vector de infectados
    #S el vector de entidades susceptibles
    #El ultimo contagia, podrías cambiar el beta y gamma posiblemente
    dt = int(duracion/c_tiempo)
    M_T = np.matrix(np.zeros((dt,dt)))
    M_T[0,0]=1
    Tc = []

    tc = c_tiempo
    
    Tc.append(c_tiempo)
    
    I = [infectado]
    i = [len(I)]
    S = [i[0] for i in poblacion.columns]
    
    s = [len(S)]
    
    S.pop(S.index(I[0]))
    
    # Se asume beta homogenio como la capacidad de infeccion fuerte
    # Sobre los demas
    
    for t in range (dt):
        
        for c in range(i[t]):
            
            p1=(beta*s[-1]*i[-1]/len(poblacion))*tc
            
            q = 1-p1
            
            u = np.random.uniform(0, 1, 1)
        
            u = u[0]

            if 0 < u and u <= p1:

                i.append(i[-1]+1)
                #I.append() aplicar un choice
            
                s.append(s[-1]-1)
            

        #elif p1 < u and u <= (p1+p2):

            #I[t+1] = I[t]-1
            
            #S[t+1] = S[t]+1          

            else:

                i.append(i[-1])
                s.append(s[-1])
            
            c += 1
            Tc.append(Tc[-1])
        Tc[-1] = Tc[-1]+c_tiempo
            
            
    
    return ([i,s,pd.DataFrame(M_T).transpose(),Tc])

def int_M_C(array):

    M_C = np.mean(array, axis=0)

    return(M_C)

def ver(array,el):

    if not el in array:

        array.append(el)

    return(array)

def union(arr1,arr2):
    
    arr = arr1*arr2
    
    return(arr)

def make_gameboard(Tamanio, JA):

    matriz = np.zeros((Tamanio,Tamanio))
    estados = ["S", "I"]
    estados2 = [f"a{i}" for i in range(Tamanio-len(estados))]
    estados3 = estados+estados2
    Matriz = pd.DataFrame(matriz)
    Matriz.loc[0,1] = 1
    for i in range(1,Tamanio-1):
        

        Matriz.loc[i, 0] = 1-JA
    
    for j in range(1, Tamanio-1):

        Matriz.loc[j, j+1] = JA

    Matriz.loc[j+1, 0] = 1

    Matriz.index = estados3
    Matriz.columns = estados3

    return(Matriz)

def get_rows(M_caracteristicas,cadena=""):
    
    if cadena == "":
        
        rows = [i[0] for i in 
            M_caracteristicas[M_caracteristicas.iloc[:, 1]!=0].index]
    
    elif type(cadena) == type([]):
        
        if len(cadena) == 2:
            
            rows = [i[0] for i in 
            M_caracteristicas[(M_caracteristicas.iloc[:, 1]==cadena[0]) | (M_caracteristicas.iloc[:, 1]==cadena[1])].index]
        
        elif len(cadena) == 3:
            
            rows = [i[0] for i in 
            M_caracteristicas[(M_caracteristicas.iloc[:, 1]==cadena[0]) | (M_caracteristicas.iloc[:, 1]==cadena[1]) | (M_caracteristicas.iloc[:, 1]==cadena[2])].index]
            
        elif len(cadena) == 4:
            
            rows = [i[0] for i in 
            M_caracteristicas[(M_caracteristicas.iloc[:, 1]==cadena[0])
                              | (M_caracteristicas.iloc[:, 1]==cadena[1]) 
                              | (M_caracteristicas.iloc[:, 1]==cadena[2])
                              | (M_caracteristicas.iloc[:, 1]==cadena[3])].index]
            
        elif len(cadena) == 5:
            
            rows = [i[0] for i in 
            M_caracteristicas[(M_caracteristicas.iloc[:, 1]==cadena[0])
                              | (M_caracteristicas.iloc[:, 1]==cadena[1]) 
                              | (M_caracteristicas.iloc[:, 1]==cadena[2])
                              | (M_caracteristicas.iloc[:, 1]==cadena[3])
                              | (M_caracteristicas.iloc[:, 1]==cadena[4])].index]
        else:
            
            rows = [i[0] for i in M_caracteristicas[(M_caracteristicas.iloc[:,1]==cadena[0])].index]
                                 
    #Ver como funciona para distintos or
    else:
        
        rows = [i[0] for i in 
            M_caracteristicas[M_caracteristicas.iloc[:, 1]==cadena].index]
    
    return(rows)


#def regladetres(dift, dif, prob):

 #   prox = prob*dif/dift

  #  return(prox)

def diferencias(Matriz, coeficientes, pesos_fila):
    
    #Evalue si los coeficientes de cada fila son superiores o inferiores al coeficiente de los otros
    #Si es inferior guardar en inf la resta del coeficiente i y su inferior
    #Si es superior guardar en sup la resta del superior y su coeficiente i
    #Divida la longitud del vector inf sobre el peso_fila i y guardelo en prob_E
    #Divida la longitud del vector sup sobre el peso_fila i y guardelo en prob_F

    Matriz2 = copy.deepcopy(Matriz) #Copie la matriz
    Matriz3 = {}
    inf = {} #Cree un diccionario de nombre inf
    sup = {} #Cree un diccionario de nombre sup
    prob_E = {} #Cree un diccionario de probabilidad de exito
    prob_F = {} #Cree un diccionario de probabilidad de fracaso
    peso_E = {}
    peso_F = {}
    i = 0
    #Usar el i de arriba para buscar las entidades
    for rows in Matriz.index: #Para toda fila en los indices de la matriz haga
        
        rows = rows[0] 
    
        inf.update({rows : {}}) #Añada un diccionario con llave la fila en inf
        sup.update({rows : {}}) #Añada un diccionario con llave la fila en sup

        for cols in Matriz.columns: #Para toda columna en los indices de la matriz haga

            cols = cols[0]
            
            c = Matriz.loc[rows,cols].to_numpy()[0][0]

            if c != 0: #Si la posición i,j es distinta de 0 haga
                
                d = coeficientes.loc[rows, :].to_numpy()[0][0]

                if c < d: #Si la posición i,j es menor o igual a un coeficiente realice
                
                    a = coeficientes.loc[rows,:].to_numpy()[0][0] - Matriz.loc[rows,cols] #Restar el JA-Jij
                    a = a.to_numpy()[0]
                    Matriz2.loc[rows,cols] = a #Guardelo en la posición i,j de la matriz
                    inf[rows].update({cols : a}) #Guardelo en el diccionario

                elif c >= d: #Si la posición i,j es mayor a un coeficiente realice
                    
                    a = Matriz.loc[rows,cols] - coeficientes.loc[rows,:].to_numpy()[0][0] #Restar el Jij-JA
                    a = a.to_numpy()[0]
                    Matriz2.loc[rows,cols] = a #Guardelo en la posición i,j de la matriz
                    sup[rows].update({cols : a}) #Guardelo en el diccionario
            
            else: 
                1+1
        
        
        prob_E.update({rows:len(inf[rows])/pesos_fila[i]})
        prob_F.update({rows:len(sup[rows])/pesos_fila[i]})
        peso_E.update({rows:np.sum([inf[rows][i] for i in inf[rows]])})
        peso_F.update({rows:np.sum([sup[rows][i] for i in sup[rows]])})
        
        i += 1

        #Idea, volver a copiar el mismo bucle y solo cambiar el valor por la prob_E/F / pesototal_E/F por el valor que ya tiene la matriz
    
    for rows in Matriz.index: #Para toda fila en los indices de la matriz haga
        
        rows = rows[0]
        Matriz3.update({rows: {"Cooperacion fuerte": inf[rows],
                               "Cooperacion debil" : sup[rows]} })
        for cols in Matriz.columns: #Para toda columna en los indices de la matriz haga

            cols = cols[0]
            
            c = Matriz.loc[rows,cols].to_numpy()[0][0]

            if c != 0: #Si la posición i,j es distinta de 0 haga
                
                d = coeficientes.loc[rows, :].to_numpy()[0][0]

                if c < d: #Si la posición i,j es menor o igual a un coeficiente realice
                
                    a = coeficientes.loc[rows,:].to_numpy()[0][0] - Matriz.loc[rows,cols] #Restar el JA-Jij
                    a = a.to_numpy()[0][0]
                    if peso_E[rows] == 0:
                        Matriz2.loc[rows,cols]=prob_E[rows]
                    else:
                        Matriz2.loc[rows,cols] = a*prob_E[rows]/peso_E[rows] #Guardelo en la posición i,j de la matriz
                        
                elif c >= d: #Si la posición i,j es mayor a un coeficiente realice
                    
                    a = Matriz.loc[rows,cols] - coeficientes.loc[rows,:].to_numpy()[0][0] #Restar el Jij-JA
                    a = a.to_numpy()[0][0]
                    if peso_F[rows] == 0:
                        Matriz2.loc[rows,cols]=prob_F[rows]
                    else:
                        Matriz2.loc[rows,cols] = a*prob_F[rows]/peso_F[rows] #Guardelo en la posición i,j de la matriz
    
            
    return([Matriz2, prob_E, prob_F, inf, sup, Matriz3])

def diferencias2(Matriz, coeficientes, pesos_fila):
    
    #Evalue si los coeficientes de cada fila son superiores o inferiores al coeficiente de los otros
    #Si es inferior guardar en inf la resta del coeficiente i y su inferior
    #Si es superior guardar en sup la resta del superior y su coeficiente i
    #Divida la longitud del vector inf sobre el peso_fila i y guardelo en prob_E
    #Divida la longitud del vector sup sobre el peso_fila i y guardelo en prob_F

    Matriz2 = copy.deepcopy(Matriz) #Copie la matriz

    inf = {} #Cree un diccionario de nombre inf
    sup = {} #Cree un diccionario de nombre sup
    prob_E = {} #Cree un diccionario de probabilidad de exito
    prob_F = {} #Cree un diccionario de probabilidad de fracaso
    peso_E = {}
    peso_F = {}
    i = 0
    #Usar el i de arriba para buscar las entidades
    for rows in Matriz.index: #Para toda fila en los indices de la matriz haga
        
        rows = rows[0] 
    
        inf.update({rows : {}}) #Añada un diccionario con llave la fila en inf
        sup.update({rows : {}}) #Añada un diccionario con llave la fila en sup

        for cols in Matriz.columns: #Para toda columna en los indices de la matriz haga

            cols = cols[0]
            
            c = Matriz.loc[rows,cols].to_numpy()[0][0]

            if c != 0: #Si la posición i,j es distinta de 0 haga
                
                d = coeficientes.loc[rows, :].to_numpy()[0][0]

                if c <= d: #Si la posición i,j es menor o igual a un coeficiente realice
                
                    a = coeficientes.loc[rows,:].to_numpy()[0][0] - Matriz.loc[rows,cols] #Restar el JA-Jij
                    a = a.to_numpy()[0]
                    Matriz2.loc[rows,cols] = a #Guardelo en la posición i,j de la matriz
                    inf[rows].update({cols : a}) #Guardelo en el diccionario

                elif c > d: #Si la posición i,j es mayor a un coeficiente realice
                    
                    a = Matriz.loc[rows,cols] - coeficientes.loc[rows,:].to_numpy()[0][0] #Restar el Jij-JA
                    a = a.to_numpy()[0]
                    Matriz2.loc[rows,cols] = a #Guardelo en la posición i,j de la matriz
                    sup[rows].update({cols : a}) #Guardelo en el diccionario
            
            else: 
                1+1
        
        
        prob_E.update({rows:len(inf[rows])/pesos_fila[i]})
        prob_F.update({rows:len(sup[rows])/pesos_fila[i]})
        peso_E.update({rows:np.sum([inf[rows][i] for i in inf[rows]])})
        peso_F.update({rows:np.sum([sup[rows][i] for i in sup[rows]])})
        
        i += 1

        #Idea, volver a copiar el mismo bucle y solo cambiar el valor por la prob_E/F / pesototal_E/F por el valor que ya tiene la matriz
    
    for rows in Matriz.index: #Para toda fila en los indices de la matriz haga
        
        rows = rows[0]
      
        for cols in Matriz.columns: #Para toda columna en los indices de la matriz haga

            cols = cols[0]
            
            c = Matriz.loc[rows,cols].to_numpy()[0][0]

            if c != 0: #Si la posición i,j es distinta de 0 haga
                
                d = coeficientes.loc[rows, :].to_numpy()[0][0]

                if c <= d: #Si la posición i,j es menor o igual a un coeficiente realice
                
                    a =  Matriz.loc[rows,cols] #Restar el JA-Jij
                    a = a.to_numpy()[0][0]
                    b = coeficientes.loc[rows,:].to_numpy()[0][0]
                    if peso_E[rows] == 0:
                        Matriz2.loc[rows,cols]=prob_E[rows]
                    else:
                        Matriz2.loc[rows,cols] = a*prob_E[rows]/b #Guardelo en la posición i,j de la matriz
                        
                elif c > d: #Si la posición i,j es mayor a un coeficiente realice
                    
                    a = Matriz.loc[rows,cols]  #Restar el Jij-JA
                    a = a.to_numpy()[0][0]
                    b = coeficientes.loc[rows,:].to_numpy()[0][0]
                    if peso_F[rows] == 0:
                        Matriz2.loc[rows,cols]=prob_F[rows]
                    else:
                        Matriz2.loc[rows,cols] = a*prob_F[rows]/b #Guardelo en la posición i,j de la matriz
    
            
    return([Matriz2, prob_E, prob_F, inf, sup])

# Python program for Dijkstra's single
# source shortest path algorithm. The program is
# for adjacency matrix representation of the graph
# Creditos a https://www.geeksforgeeks.org/python-program-for-dijkstras-shortest-path-algorithm-greedy-algo-7/
class Graph():
 
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]
       
 
    def printSolution(self, dist):
        columns = ["Distancias"]
        nodes = []
        dists = []
        for node in range(self.V):
            nodes.append(node)
            dists.append(dist[node])
        self.frame = pd.DataFrame(dists, index = nodes, columns=columns)
 
    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minDistance(self, dist, sptSet):
 
        # Initialize minimum distance for next node
        min = 1e7
 
        # Search not nearest vertex not in the
        # shortest path tree
        for v in range(self.V):
            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                min_index = v
                
 
        return min_index
 
    # Function that implements Dijkstra's single source
    # shortest path algorithm for a graph represented
    # using adjacency matrix representation
    def dijkstra(self, src):
        
        dist = [1e7] * self.V
        dist[src] = 0
        sptSet = [False] * self.V
        rutas = {}
        for cout in range(self.V):
            
            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.minDistance(dist, sptSet)
 
            # Put the minimum distance vertex in the
            # shortest path tree
            sptSet[u] = True
 
            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shortest path tree
           
            #cout es el vector de conexion
            #v es el nodo que se conecta con el nodo previo
            
            
           
            for v in range(self.V):
                
                vector = []    
                
                if (self.graph[u][v] > 0 and
                   sptSet[v] == False and
                   dist[v] > dist[u] + self.graph[u][v]):
                    dist[v] = dist[u] + self.graph[u][v]
                    #print(self.graph)
                    #print(dist[v], "distancia de v")
                    #print(dist, sptSet)
                    
                vector.append([cout,v])
                rutas.update({f"{vector}": dist[v]})
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        rutas = pd.DataFrame(rutas.values(), index = rutas.keys())
        
        #print(rutas)         
 
        self.printSolution(dist)
        
        return(self.frame)
    
 #Driver program
#g = Graph(9)
#g.graph = np.array([[0, 4, 0, 0, 0, 0, 0, 8, 0],
       #    [4, 0, 8, 0, 0, 0, 0, 11, 0],
        #   [0, 8, 0, 7, 0, 4, 0, 0, 2],
         #  [0, 0, 7, 0, 9, 14, 0, 0, 0],
          # [0, 0, 0, 9, 0, 10, 0, 0, 0],
           #[0, 0, 4, 14, 10, 0, 2, 0, 0],
           #[0, 0, 0, 0, 0, 2, 0, 1, 6],
           #[8, 11, 0, 0, 0, 0, 1, 0, 7],
           #[0, 0, 2, 0, 0, 0, 6, 7, 0]
           #])


 
#print(g.dijkstra(0))



def dijkstra(graph, start, end):
    # Creamos un diccionario para almacenar los nodos previos
    prev = {}
    # Creamos un diccionario para almacenar las distancias
    distances = {}
    # Inicializamos todas las distancias en infinito
    # y todos los nodos previos en None
    for node in graph:
        distances[node] = float('inf')
        prev[node] = None
    # Establecemos la distancia del nodo de inicio en 0
    distances[start] = 0
    # Creamos una cola de prioridad para seleccionar el nodo más cercano
    # utilizamos el módulo heapq y tuplas (distance, node)
    heap = [(0, start)]
    # Mientras haya nodos en la cola de prioridad
    while heap:
        # Tomamos el nodo más cercano
        distance, node = heapq.heappop(heap)
        # Si llegamos al nodo final, terminamos el ciclo
        if node == end:
            break
        # Si no hemos visitado aún el nodo
        if distance == distances[node]:
            # Iteramos sobre sus vecinos
            for neighbor, cost in graph[node].items():
                # Calculamos la nueva distancia al vecino
                new_distance = distance + cost
                # Si la nueva distancia es menor que la distancia actual del vecino
                if new_distance < distances[neighbor]:
                    # Actualizamos la distancia del vecino
                    distances[neighbor] = new_distance
                    # Establecemos el nodo actual como el nodo previo del vecino
                    prev[neighbor] = node
                    # Añadimos el vecino a la cola de prioridad
                    heapq.heappush(heap, (new_distance, neighbor))
    # Creamos una lista para almacenar el camino más corto
    path = []
    # Establecemos el nodo final como el nodo actual
    node = end
    # Mientras tengamos un nodo previo
    while node is not None:
        # Añadimos el nodo a la lista
        path.append(node)
        # Actualizamos el nodo actual con su nodo previo
        node = prev[node]
    # Devolvemos la lista invertida (del nodo final al nodo inicial)
    return path[::-1]

#https://chat.openai.com/chat

 
# This code is contributed by Divyanshu Mehta


#Por ultimo el ambito competitivo

def competencia(Ai, A_k, B, M, M_caracteristicas):
    
    #A_k es el vector de competidores
    #A_i es el competidor a seguir
    #B es el nodo que posee el recurso
    #A_i y B pertenecen a A_k
    #M la matriz de conexiones, con ayuda del codigo anterior
    #Nos aproximaremos a las distancias minimas necesarias,
    #Para maximizar el competitivo.
    
    g = Graph(len(A_k))
    
    M = M.to_numpy()
    
    g.graph = M
    
    D_A_i = {}
    
    for i in A_k:
    
        G = g.dijkstra(A_k.index(i))
        
        G.index = A_k
        
        D_A_ij = G.loc[B, :].to_numpy()[0] * (1/M_caracteristicas.loc[i, :].to_numpy()[0])
      
        D_A_i.update({i:D_A_ij})
        
    
    del D_A_i[B] # Excluimos B pues, el no compite sobre si mismo
    D_AI = D_A_i[Ai]
    inf = {}
    
    sup = {}
    
    for i in D_A_i:
        
        if D_A_i[i] < D_A_i[Ai]: #Posible correccion borrar el comentario anterior hasta aqui D_AI
            
            inf.update({i : D_A_i[Ai]-D_A_i[i]})
            
        else:
            
            sup.update({i : D_A_i[i]-D_A_i[Ai]})
            
    prob_com_debil = len(inf)/len(D_A_i) # Si mi forma de llegar 
    #Al recurso es mas larga que la de la competencia, entonces
    #Sere un competidor debil
    
    prob_com_fuerte = len(sup)/len(D_A_i)# Caso contrario,
    #Tendre una ventaja al competir, esta ventanja se puede mejorar
    #Por medio de inyectar capacidades
    
    #En este modelo, se puede ver tambien la necesidad de algunas,
    #Conexiones, como Min_ciencias, Lo cual asegure mayor, estabilidad
    
    return ([prob_com_fuerte, prob_com_debil, inf, sup])
            
    
        
#Metodo de aceptacion y rechazo para modificar capacidades
def estimar_beta(mean, var):
    
    alpha = mean**2 - mean**3 - mean*var
    alpha = alpha/var
    
    beta = mean - var + mean**3 - mean*var
    beta = beta/var
    
    return(alpha, beta)
    
    #Creo que aquí va un coeficiente de Jaccard se distribuye Beta
    #Generar la uniforme
    #Si U < la acumulada de (Y)
#Metodo de simulacion de la matriz por medio de Bernoullis



#La simulacion de coeficientes se puede hacer bajo condicionales
def new_conex(M_conex):
    
    np.random.seed(1)
    
    prob = M_conex.sum()/len(M_conex)
    Matriz2 = copy.deepcopy(M_conex)
    
    for i in prob.index:
        
        a = prob.loc[i, :]
        a = a.to_numpy()[0]
        
        Matriz2.loc[i, :] = np.random.binomial(1, 0.5, len(M_conex))
    np.random.seed(None)
    return(Matriz2)
        
        
    
    # Defino la funcion cambiar que depende del coeficiente del
    # actor infeccioso y el vector de caracteristicas seleccionado
    
def incremento(CTI, CTIS):
    
        # Se 1-CTI
    np.random.seed(1)
    CTI = CTI.to_numpy()[0]
    
    CTJ = copy.deepcopy(CTIS)
    
    x = 1-CTI
        # U = Simula uniformes para llegar a 1
    u = np.random.uniform(0, x)
        # Si U+CTIj < 1:
    for i in CTIS.index :
        
        a = CTJ.loc[i, :]
        a = a.to_numpy()[0]
        
        if u+a < 1:
        
            # CTIj = U+CTI
            
            CTJ.loc[i, :] = u+a
        
            # Lo ultimo será la probabilidad de incremento
            # en las capacidades tecnologicas
    np.random.seed(None)
    return (CTJ)
        
        
        
        
        
    
    
    
    
