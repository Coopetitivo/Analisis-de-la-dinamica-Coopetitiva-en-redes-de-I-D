import tkinter as tk
from tkinter import ttk

class SIS_interfaz(tk.Tk):

    def __init__(self):
        tk.Tk.__init__(self) #inicio la pantalla
        self.wm_title('Coopetitivo') #Atribuyo el titulo
        return(self.coop()) #Regreso la pantalla del coopetitivo


    def coop(self):
        self.cont0 = tk.Frame(self, bg="azure") #Creo un contenedor madre
        self.cont0.pack(expand=True, fill='both')
        self.container = tk.Frame(self.cont0, bg = 'azure') #Creo un contenedor para una combo
        # self.container2 = tk.Frame(self.cont, bg = 'mintcream') #Creo un contenedor para otro combo
        self.label = tk.Label(self.container,text = "Matriz a usar")
        self.label.pack()
        self.sim = tk.StringVar()
        self.sims = ttk.Combobox(self.container,
                                values = ["Simular Matriz de conexiones",
                                          "Matriz de conexiones estandar"],
                                state="readonly")
        self.sims.set("Matriz de conexiones estandar")
        self.sims.pack(pady=10, padx=10)        
           
        self.coop1 = ttk.Combobox(self.container,
                values = ["Competitivo",
                    "Cooperativo", "Coopetitivo"],
                state = "readonly") #Combo para que el usuario escoja si desea analizar la parte cooperativa, competitiva o coopetitiva

        self.coop1.set("Cooperativo")
        self.coop1.pack(pady=10, padx=10)

        #self.coop2 = ttk.Combobox(self.container2,
         #       values = ["Vertical", "Horizontal"],
          #      state = "readonly") #Combo para que el usuario escoja si desea analizar la parte vertical u horizontal
        #self.coop2.set('Vertical')
        #self.coop2.pack(pady=10, padx=10)

        self.button = tk.Button(self.cont0, text = 'Siguiente',
                command=lambda : [self.coopetitivo(),
                    self.cont0.destroy()], bg = 'mintcream')

        self.button.pack(side = "bottom",pady = 10 , padx = 10)

        self.container.pack(fill='both',
                side='left', expand=True)
        #self.container2.pack(fill='both' ,
         #       side='right' ,expand = True)

    def coopetitivo(self):
        
        self.coop1 = self.coop1.get()#Rescato la variable antes de que su contenedor sea destruido
        self.sim.set(self.sims.get())
        #no es necesario crear tres funciones para cumplir lo mismo
        #por ende, lo unico que necesitamos es una boton especificio
        #que ayude a maximizar las funcionales
        #Por ende, lo unico que nos termina por interezar es el nodo
        #de infeccion que se soluciona con poner despues de pasar un
        #nodo B que nos ayude

        ##completar

        #Crear un contenedor para tres opciones

        self.cont = tk.Frame(self, bg="azure")
        self.cont.pack(fill="both", expand=True)
#################
        self.container1 = tk.Frame(self.cont, bg = 'azure')
        self.container1.pack(side='left', expand=True, fill="both")
        #Completar el container 1: Va a agrupar las caracteristicas en casos horizontales/verticales
        self.labc = tk.Label(self.container1,text="Seleccione las CTI de necesidad")

        self.labc.pack()


        vals = list(caracteristica.values())[2:6]#Capacidades

        self.cti = []

        self.carac = ttk.Combobox(self.container1,
                values = vals, state = "readonly")

        self.carac.set(vals[0])

        self.carac.pack(padx=10, pady=10)

        self.botton = tk.Button(self.container1,
                text = "Agregar capacidad",
                bg="azure",
                command = lambda : [self.cti.append(self.carac.get())])

        self.botton.pack(padx=10, pady=10)

#################
        self.container2 = tk.Frame(self.cont, bg = 'azure')
        self.container2.pack(side='right', expand=True, fill='both')
        #Completar el container 2: Va a agrupar los multiples tipos

        vals2 = np.unique(list(M_caracteristicas.iloc[:, 1]))
        vals2 = list(vals2)

        self.labc1 = tk.Label(self.container2,
                text="Seleccione el tipo de actores")

        self.labc1.pack()

        self.tip = []

        self.tipos = ttk.Combobox(self.container2,
                values = vals2,
                state = "readonly")

        self.tipos.set(vals2[0])

        self.tipos.pack(padx=10, pady=10)

        self.botton2 = tk.Button(self.container2,text="Agregar tipo de actor",
                bg="azure",
                command = lambda : [self.tip.append(self.tipos.get())])

        self.botton2.pack(padx=10, pady=10)
#################
        self.container3 = tk.Frame(self.cont, bg = 'azure')
        self.container3.pack(side='bottom', expand=True, fill='both')

        self.var = ttk.Combobox(self.container3,
                values=["Pseudoaleatorio","Manual"],
                state = "readonly")

        self.var.set("Manual")
        self.var.pack(padx=10, pady=10)

        self.botton3 = tk.Button(self.container3,text="Siguiente",
                bg="azure",
                command = lambda:[self.pasar1(), self.cont.destroy()])
        self.botton3.pack()
        #Crear una caja para las capacidades y un boton de aniadir
        #Crear una caja para los tipos de actor y un boton de aniadir
        #Crear un boton para trabajar o pseudoaleatorio/manual
        
    def pasar1(self):

        self.tipos = np.unique(self.tip)
        self.rows = Sc.get_rows(M_caracteristicas, self.tip)
        #Nombres segun el tipo
        if self.sim.get() == "Simular Matriz de conexiones":
            self.matriz = (Sc.new_conex(M_conex)).loc[self.rows, self.rows]
            self.matriz.to_csv("out/Matrizdeactores.csv", sep=";")
        else:    
            self.matriz = M_conex.loc[self.rows, self.rows]
        #Valores a buscar en la matriz
        self.tiempo = 0.1
        self.duracion = 300
        
        if self.var.get() == "Manual":
            self.cont3 = tk.Frame(self, bg="azure")
            self.cont3.pack(fill="both", expand=True)
            self.choice = tk.StringVar()
            self.choices = ttk.Combobox(self.cont3,
                    values=self.rows,
                    state = "readonly")
            self.choices.set(self.rows[0])
            self.choices.pack(padx=10, pady=10)
            self.bot = tk.Button(self.cont3,
                                 text="Selecionar actor infecioso",
                    bg="azure",
                    command = lambda:[self.choice.set(self.choices.get()),
                                      self.pasar2(),self.cont3.destroy()])
            self.bot.pack(padx=10,pady=10)
            self.var = self.var.get()

        else:

            self.choice = tk.StringVar()

            a = np.random.choice(self.rows,1)

            self.choice.set(a[0])
            
            self.var = self.var.get()
            
            return(self.pasar2())
        
    def pasar2(self):
        
        mat_power = Sc.mat_power(1,self.matriz)
        self.mat_power = mat_power
        self.poblacion = (mat_power.loc[self.choice.get(),
                                   self.rows]!=0).sum().sum() + 1
        
        if self.coop1 != "Cooperativo":
            
            return(self.pasar3())

        else:

            return(self.cooperativo_())

    def cooperativo_(self):

        #Aqui iran los calculos del cooperativo
        self.EX_FA={}
        self.T = {}
        self.sis = {}
        
        for i in self.cti:

            cof = M_caracteristicas.loc[self.rows, i]
            cof2 = M_caracteristicas.loc[self.choice.get(), i]
            Matriz = Sc.diferencias(Sc.make_jaccard(self.matriz,
                cof), cof, self.matriz.sum())
            
            self.Matriz_ = Matriz[0]

            ex_fa = [Matriz[1][self.choice.get()],
                    Matriz[2][self.choice.get()]]
            self.EX_FA.update({i:ex_fa})

            beta = ex_fa[0]

            tiempo_promedio = 1/(1-beta)

            self.T.update({i:tiempo_promedio})

            mod_sis = Sc.DTMC_SIS(beta,
                    1-beta, self.poblacion,
                    self.tiempo,
                    self.duracion)

            self.y0 = []
            self.y1 = []
            self.t = []

            mat_p = Sc.make_prob(Matriz[0])
            mat_p.to_csv(f"out/Matrizt {i}.csv", sep=";")
            Sc.mat_power(10, mat_p).to_csv(f"out/tpasos {i}.csv", sep=';')

            for j in range(30):

                mod_sis2 = Sc.DTMC_SIS(beta, 1-beta,
                        self.poblacion, self.tiempo,
                        self.duracion)
                self.t = mod_sis[-1]
                self.y0.append(mod_sis2[0])
                self.y1.append(mod_sis2[1])
            
            self.sis.update({i : [Sc.int_M_C(self.y0),
                                  Sc.int_M_C(self.y1)]})
            #Aqui termina la cooperacion, por ende, aqui se simula
            #Las capacidades
            print(cof)
            cat = Sc.incremento(cof2, cof)
            print(cat)
                
        return(self.graph())
            
    def pasar3(self):
        
        self.rows2 = copy.deepcopy(self.rows)
        self.rows2.pop(self.rows.index(self.choice.get()))
        
        if self.var == "Manual":
            
            self.container = tk.Frame(self, bg = "azure")
            self.container.pack(expand=True, fill="both")
            
            self.b = ttk.Combobox(self.container, values = self.rows2,
                         state="readonly")
            self.b.set(self.rows2[0])
            self.b.pack()
            button = tk.Button(self.container,
                           text="Selecionar actor susceptible",
                           bg="azure")
            
            if self.coop1 == "Coopetitivo":
                
                button.config(command=lambda:[self.coopetitivo_(),
                                              self.container.destroy()])
                
                button.pack()
                
            else:
                
                button.config(command=lambda:[self.competitivo_(),
                                              self.container.destroy()])
                button.pack()
        
        else:
            
            self.b = tk.StringVar()
            self.b.set(np.random.choice(self.rows2))
            
            if self.coop1 == "Coopetitivo":
                
                return(self.coopetitivo_())
            
            else:
                
                return(self.competitivo_())
        
        

    def competitivo_(self):

        #Escoja dentro de las filas,
        #un valor diferente al escogido en choice de forma aleatoria
        #Para que la competicion sea lo mas justa
        b = self.b.get()
        self.rows3 = copy.deepcopy(self.rows)
        self.rows3.pop(self.rows.index(b))
        self.EX_FA = {}
        self.sis = {}
        
        for k in self.cti:
            
            self.Matriz_ = self.matriz
            
            for i in self.rows3:

                cof = M_caracteristicas.loc[self.rows, self.cti[0]]

                self.comp = Sc.competencia(i,
                    self.rows, b, self.matriz,
                    cof)

                self.EX_FA.update({f"{k} {i}":self.comp[0:2]})

                self.y0 = []
                self.y1 = []
                self.t = []

                for j in range(30):

                    md = Sc.DTMC_SIS(self.EX_FA[f"{k} {i}"][0],
                                     1-self.EX_FA[f"{k} {i}"][0],
                                     len(self.rows3),
                        #len (self.rows3) hace referencia a la poblacion
                        # Sin embargo esta esta sesgada
                        # La mejor forma sería calcular mat_power
                        # Y ver en cuantos es distinto a 0 
                        # Despues del proceso.
                            self.tiempo, self.duracion)

                    self.y0.append(md[0])
                    self.y1.append(md[1])
                    self.t = md[-1]

                self.sis.update({f"{k} {i}":[Sc.int_M_C(self.y0),
                                    Sc.int_M_C(self.y1)]})

            #Lo siguiente es agregar las graficas multiples

        #Aqui iran los calculos del competitivo
            
        return(self.graph())
        
    def coopetitivo_(self):

        self.EX_FA = {}
        b = self.b.get()
        self.rows3 = copy.deepcopy(self.rows)
        self.rows3.pop(self.rows.index(b))
        self.EX_FA = {}
        self.sis = {}

        for i in self.cti:

            cof = M_caracteristicas.loc[self.rows, i]
            
            Matriz = Sc.diferencias(Sc.make_jaccard(self.matriz, cof),
                    cof, self.matriz.sum())
            self.Matriz_ = Matriz[0]
            #print(Matriz[0])

            #Matriz = Sc.diferencias2(Sc.make_jaccard(self.matriz, cof),
             #       cof, self.matriz.sum())
            
            #print(Matriz[0])

            mat_p = Sc.make_prob(Matriz[0])
            
            #Aqui hacen cooperacion por ende en este punto
            #Se deben de simular capacidades
            cof4 = cof
            for j in self.rows3:
                
                cof3 = M_caracteristicas.loc[j, i]
                cof4 = Sc.incremento(cof3, cof4)

            for j in self.rows3:
                
                self.poblacion = (self.mat_power.loc[j,
                                   self.rows]!=0).sum().sum() + 1

                #cof2 = M_caracteristicas.loc[self.rows, i]
                
                self.comp = Sc.competencia(j, #Falla el sistema al buscar algunos indices / Solucionando
                        self.rows, b, mat_p, cof4)

                self.EX_FA.update({f"{i} {j}": self.comp[0:2]})

                self.y0 = []
                self.y1 = []
                self.t = []

                for k in range(30):

                    md = Sc.DTMC_SIS(self.EX_FA[f"{i} {j}"][0],
                            1-self.EX_FA[f"{i} {j}"][0],
                            self.poblacion,
                            self.tiempo, self.duracion)

                    self.y0.append(md[0])
                    self.y1.append(md[1])
                    self.t = md[-1]

                self.sis.update({f"{i} {j}":[Sc.int_M_C(self.y0),
                    Sc.int_M_C(self.y1)]})

        #Aqui iran los calculos del competitivo
        return(self.graph())
    
    def graph(self):
        
        ##################Parte modelo sis
        
        self.cont.destroy()
    
        self.cont0 = tk.Frame(self, bg="azure")
        self.container1 = tk.Frame(self.cont0, bg = "azure")
        self.container2 = tk.Frame(self.cont0, bg = "azure")
    
        self.cont0.pack(expand=True, fill="both")
        self.container1.pack(expand=True, fill="both", side="left")
        self.container2.pack(expand=True, fill="both", side="right")
    
        self.fig = mtp.figure.Figure()
        self.si = self.fig.add_subplot(111)

        for i in self.sis:
            
            #longitud = len(self.matriz.to_numpy()[0])
            
            self.si.set_xlabel("Tiempo")
            self.si.set_ylabel("Densidad de la problación")
            a = self.coop1 != "Cooperativo"
            b = True #self.choice.get() in i
            
            if (a,b) == (True, True):
                self.si.set_title(f"{self.b.get()}")
                self.si.plot(self.t, (self.sis[i][0]/self.poblacion),
                         label = i + " " + r"$\beta = $"+ f"{np.round(self.EX_FA[i][0],3)}")
            elif a == False:    
                self.si.set_title(f"{self.choice.get()}")
                self.si.plot(self.t, (self.sis[i][0]/self.poblacion),
                         label = i + " " + r"$\beta = $"+ f"{np.round(self.EX_FA[i][0],3)}")
        
        self.si.legend()
            
        canvas0 = FigureCanvasTkAgg(self.fig, master = self.container2)
        toolbar = NavigationToolbar2Tk(canvas0, self.container2)
        canvas0.draw()
        canvas0.get_tk_widget().pack(side = "right",
                                    expand=True, fill="both")
        canvas0._tkcanvas.pack(side="right",
                               fill="both", expand=True)
            
        
       
        ###################Parte grafo
        self.fig2 = mtp.figure.Figure()
        self.gra = self.fig2.add_subplot(111)
    
        Sc.graph(np.round(self.Matriz_.loc[self.rows, self.rows],3),
                 M_caracteristicas.iloc[:,1])
        plt.savefig("out/"+"Grafo de actores involucrados"+".png", bbox_inches="tight", dpi=1200)
        plt.close()
        
        img_arr = mtp.image.imread("out/"+"Grafo de actores involucrados"+".png")
        self.gra.imshow(img_arr)
        self.gra.set_title("Grafo de actores involucrados")
        
        self.gra.spines["top"].set_visible(False)
        self.gra.spines["left"].set_visible(False)
        self.gra.spines["right"].set_visible(False)
        self.gra.spines["bottom"].set_visible(False)
        self.gra.xaxis.set_visible(False)
        self.gra.yaxis.set_visible(False)
        
        canvas = FigureCanvasTkAgg(self.fig2, master = self.container1)
        toolbar = NavigationToolbar2Tk(canvas, self.container1)
        canvas.draw()
        canvas.get_tk_widget().pack(side="left", fill="both", expand=True)
        canvas._tkcanvas.pack(side="left", fill="both", expand=True)
        

if __name__ == "__main__":
    
    import pandas as pd
    pd.set_option("display.max_rows", 5)
    pd.set_option("display.max_columns", 5)
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg 
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
    import matplotlib as mtp
    import copy
    import Simulacion_coopetitivo as Sc
    
##################Globalizacion solo para agilizar

    global caracteristica
    global M_caracteristicas
    global M_conex
    global coeficientes
    global pesos
    global pef

     
##################Diccionario de datos

    caracteristica = {"1":"Nro de topicos","2":"Tipo de actor", "3":"Capacidad de investigacion",
"4":"Capacidad de desarrollo", "5":"Capacidad de difusion", "6":"Capacidad de mercadeo"}

#################

################# Matriz de caracteristicas y conexiones

    M_caracteristicas = pd.read_csv("in/Matriz de CTI-ok.csv",sep=",",
                                header=None, index_col=0, decimal=",")
    M_caracteristicas.columns=[list(caracteristica.values())]

    entidades = {f"{i+1}":
             M_caracteristicas.index[i] for i in range(len(M_caracteristicas.index))}

    M_caracteristicas.reset_index(drop=True, inplace=True)

    M_caracteristicas.index=[list(entidades.values())]

    M_conex = pd.read_csv("in/matrizactores.csv", sep=",", header=None)

    M_conex.columns=[list(entidades.values())]

    M_conex.index=[list(entidades.values())]

    M_conex = Sc.corrector(M_conex)
    
    #M_conex = M_conex.loc[M_conex.index != ('CIAT',),
        #                  M_conex.columns != ('CIAT',)]
    #M_conex = M_conex.loc[M_conex.index != ('UNIVALLE',),
       #                   M_conex.columns != ('UNIVALLE',)]
    #M_conex = M_conex.loc[M_conex.index != ('UNAL',),
      #                    M_conex.columns != ('UNAL',)]
    
    #M_caracteristicas = M_caracteristicas.loc[M_caracteristicas.index != ('CIAT',),
     #                     M_caracteristicas.columns != ('CIAT',)]
    #M_caracteristicas = M_caracteristicas.loc[M_caracteristicas.index != ('UNIVALLE',),
     #                     M_caracteristicas.columns != ('CIAT',)]
    #M_caracteristicas = M_caracteristicas.loc[M_caracteristicas.index != ('UNAL',),
      #                    M_caracteristicas.columns != ('CIAT',)]
################## Coeficientes

    coeficientes = M_caracteristicas[M_caracteristicas.columns[-4:]]

################# Pesos

    pesos = []        
    pef = []

    for i in M_conex.columns:

        pf = M_conex.transpose()[i].sum()

        pef.append(pf)

        pesos.append(2*pf)
        
    #a = Sc.estimar_beta(np.mean(M_caracteristicas.loc[Sc.get_rows(M_caracteristicas, cadena="Difusor"),
#"Capacidad de investigacion"]), np.var(M_caracteristicas.loc[Sc.get_rows(M_caracteristicas, cadena="Difusor"),
#"Capacidad de investigacion"]))
    #b = np.random.beta(a[0], a[1], size=len(Sc.get_rows(M_caracteristicas, cadena="Difusor")))
    
    root = SIS_interfaz()
    root.mainloop()
    
    #Existe alguna forma de cambiar alguno de los datos sin alterar
    #La interfaz?


