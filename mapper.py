import sys
import random
import numpy
import math
import StringIO
import time
from __future__ import division


# def main():
#     # sys.stdin = StringIO.StringIO(test_text)
#     mapper()
#     sys.stdin = sys.__stdin__

class SMO():
    def __init__(self,lb1,ub1,dim1,PopSize1,acc_err1,iters1):
        self.PopSize=PopSize1
        self.dim=dim1
        self.acc_err=acc_err1
        self.lb=lb1
        self.ub=ub1
        self.pos=numpy.zeros((PopSize1,dim1))
        self.fun_val = numpy.zeros(PopSize1)
        self.fitness = numpy.zeros(PopSize1)
        self.gpoint = numpy.zeros((PopSize1,2))
        self.prob=numpy.zeros(PopSize1)
        self.LocalLimit=dim1*PopSize1;
        self.GlobalLimit=PopSize1;
        self.fit = numpy.zeros(PopSize1)
        self.MinCost=numpy.zeros(iters1)
        self.Bestpos=numpy.zeros(dim1)
        self.group = 0
        self.func_eval=0
        self.part=1
        self.max_part=5
        self.cr=0.1

    def objf(self,x):
        return numpy.sum(x**2)
    
    def CalculateFitness(self,fun1):
        if fun1 >= 0:
            result = (1/(fun1+1))
        else:
            result=(1+math.fabs(fun1))
        return result
    
    
    def initialize(self):
        global GlobalMin, GlobalLeaderPosition, GlobalLimitCount, LocalMin, LocalLimitCount, LocalLeaderPosition
        S_max=int(self.PopSize/2)
        LocalMin = numpy.zeros(S_max)
        LocalLeaderPosition=numpy.zeros((S_max,self.dim))
        LocalLimitCount=numpy.zeros(S_max)
        for i in range(self.PopSize):
            for j in range(self.dim):
                if type(self.ub)==int:
                    self.pos[i,j]=random.random()*(self.ub-self.lb)+self.lb
                else:
                    self.pos[i,j]=random.random()*(self.ub[j]-self.lb[j])+self.lb[j]
        
        for i in range(self.PopSize):
            # Performing the bound checking
            self.pos[i,:]=numpy.clip(self.pos[i,:], self.lb, self.ub)
            self.fun_val[i]=self.objf(self.pos[i,:])
            self.func_eval+=1
            self.fitness[i]=self.CalculateFitness(self.fun_val[i])

        
        GlobalMin=self.fun_val[0]
        GlobalLeaderPosition=self.pos[0,:]
        GlobalLimitCount=0

        
        for k in range(self.group):
            LocalMin[k]=self.fun_val[int(self.gpoint[k,0])]
            LocalLimitCount[k]=0
            LocalLeaderPosition[k,:]=self.pos[int(self.gpoint[k,0]),:]

    def CalculateProbabilities(self):
        maxfit=self.fitness[0];
        i=1
        while(i<self.PopSize):
            if (self.fitness[i]>maxfit):
                maxfit=self.fitness[i];
            i+=1
        for i in range(self.PopSize):
            self.prob[i]=(0.9*(self.fitness[i]/maxfit))+0.1;
            
            
    def create_group(self):
        g=0
        lo=0
        while(lo < self.PopSize):
            hi= lo+int(self.PopSize/self.part)
            self.gpoint[g,0]=lo
            self.gpoint[g,1]=hi
            if((self.PopSize-hi)<(int(self.PopSize/self.part))):
                self.gpoint[g,1]=(self.PopSize-1)
            g=g+1
            lo=hi+1
        self.group = g
        
        
    def LocalLearning(self):
        global LocalMin, LocalLimitCount, LocalLeaderPosition
        S_max=int(self.PopSize/2)
        OldMin = numpy.zeros(S_max)
        for k in range(self.group):
            OldMin[k]=LocalMin[k]

        for  k in range(self.group):
            i=int(self.gpoint[k,0])
            while (i<=int(self.gpoint[k,1])):
                if (self.fun_val[i]<LocalMin[k]): 
                    LocalMin[k]=self.fun_val[i]
                    LocalLeaderPosition[k,:]=self.pos[i,:]
                i=i+1
       
        for k in range(self.group):
            if (math.fabs(OldMin[k]-LocalMin[k])<self.acc_err):
                LocalLimitCount[k]=LocalLimitCount[k]+1
            else:
                LocalLimitCount[k]=0
                
    def GlobalLearning(self):
        global GlobalMin, GlobalLeaderPosition, GlobalLimitCount
        G_trial=GlobalMin
        for i in range(self.PopSize):
            if (self.fun_val[i] < GlobalMin):
                GlobalMin=self.fun_val[i]
                GlobalLeaderPosition=self.pos[i,:]

        if(math.fabs(G_trial-GlobalMin)<self.acc_err):
            GlobalLimitCount=GlobalLimitCount+1
        else:
            GlobalLimitCount=0
            
    def LocalLeaderPhase(self,k):
        global LocalLeaderPosition
        new_position=numpy.zeros((1,self.dim))
        lo=int(self.gpoint[k,0])
        hi=int(self.gpoint[k,1])
        i=lo
        while(i <=hi):
            while True:
                PopRand=int((random.random()*(hi-lo)+lo))
                if (PopRand != i):
                    break
            for j in range(self.dim):
                if (random.random() >= self.cr):
                    new_position[0,j]=self.pos[i,j]+(LocalLeaderPosition[k,j]-self.pos[i,j])*(random.random())+(self.pos[PopRand,j]-self.pos[i,j])*(random.random()-0.5)*2
                else:
                    new_position[0,j]=self.pos[i,j]
            new_position=numpy.clip(new_position, self.lb, self.ub)
            
            ObjValSol=self.objf(new_position)
            self.func_eval+=1
            FitnessSol=self.CalculateFitness(ObjValSol)
            if (FitnessSol>self.fitness[i]):
                self.pos[i,:]=new_position
                self.fun_val[i]=ObjValSol
                self.fitness[i]=FitnessSol
            i+=1
            
            
    def GlobalLeaderPhase(self,k):
        global GlobalLeaderPosition
        new_position=numpy.zeros((1,self.dim))
        lo=int(self.gpoint[k,0])
        hi=int(self.gpoint[k,1])
        i=lo;
        l=lo;
        while(l<hi):
            if (random.random() < self.prob[i]):
                l+=1
                while True:
                    PopRand=int(random.random()*(hi-lo)+lo)
                    if (PopRand != i):
                        break
                param2change=int(random.random()*self.dim)
                new_position=self.pos[i,:]
                new_position[param2change]=self.pos[i,param2change]+(GlobalLeaderPosition[param2change]-self.pos[i,param2change])*(random.random())+(self.pos[PopRand,param2change]-self.pos[i,param2change])*(random.random()-0.5)*2
                new_position=numpy.clip(new_position, self.lb, self.ub)
                ObjValSol=self.objf(new_position)
                self.func_eval+=1
                FitnessSol=self.CalculateFitness(ObjValSol)
                if (FitnessSol>self.fitness[i]):
                    self.pos[i,:]=new_position
                    self.fun_val[i]=ObjValSol
                    self.fitness[i]=FitnessSol
            i+=1;
            if (i==(hi)):
                i=lo;
                
    def GlobalLeaderDecision(self):
        global GlobalLimitCount
        if(GlobalLimitCount> self.GlobalLimit):
            GlobalLimitCount=0
            if(self.part<self.max_part):
                self.part=self.part+1
                self.create_group()
                self.LocalLearning()
            else:
                self.part=1
                self.create_group()
                self.LocalLearning()
                
                
    def LocalLeaderDecision(self):
        global GlobalLeaderPosition, LocalLimitCount, LocalLeaderPosition
        for k in range(self.group):
            if(LocalLimitCount[k]>self.LocalLimit):
                i=self.gpoint[k,0]
                while(i<=int(self.gpoint[k,1])):
                    for j in range(self.dim):
                        if (random.random()>= self.cr):
                            if type(self.ub)==int:
                                self.pos[i,j]=random.random()*(self.ub-self.lb)+self.lb
                                self.pos[i,j]=numpy.clip(self.pos[i,j], self.lb, self.ub)
                            else:
                                self.pos[i,j]=random.random()*(self.ub[j]-self.lb[j])+self.lb[j]
                                self.pos[i,j]=numpy.clip(self.pos[i,j], self.lb, self.ub)
                        else:
                            self.pos[i,j]=self.pos[i,j]+(GlobalLeaderPosition[j]-self.pos[i,j])*random.random()+(self.pos[i,j]-LocalLeaderPosition[k,j])*random.random()
                            self.pos[i,j]=numpy.clip(self.pos[i,j], self.lb, self.ub)
                    self.pos[i,:]=numpy.clip(self.pos[i,:], self.lb, self.ub)
                    self.fun_val[i]=self.objf(self.pos[i,:])
                    self.func_eval+=1
                    self.fitness[i]=self.CalculateFitness(self.fun_val[i])
                    i+=1
                LocalLimitCount[k]=0
                
                
                
def main():
     # sys.stdin = StringIO.StringIO(test_text)
    for line in sys.stdin:
        data = line.strip().split()
        if(len(data)==5):
            PopSize, lb, ub, dim, iters = data
        else:
            continue
    
#     PopSize = int(PopSize1)
#     lb = int(lb1)
#     ub = int(ub1)
#     dim = int(dim1)
#     iters = int(iters1)

    smo=SMO(lb,ub,dim,PopSize,1.0e-5,iters)

    smo.initialize()
    
    smo.GlobalLearning()
    
    smo.LocalLearning()
    
    smo.create_group()
    
    for l in range(iters):
        for k in range(smo.group):
    
            smo.LocalLeaderPhase(k)
            
    
        smo.CalculateProbabilities()

        for k in range(smo.group):
    
            smo.GlobalLeaderPhase(k)
            
    
        smo.GlobalLearning()

        smo.LocalLearning()

        smo.LocalLeaderDecision()
    
        smo.GlobalLeaderDecision()
    
        smo.cr = smo.cr + (0.4/iters)
            
        smo.MinCost[l] = GlobalMin
        Bestpos=smo.pos[1,:]
        gBestScore=GlobalMin
        
    print(smo.objf(Bestpos)," ",str(Bestpos))
    
    
if __name__ == "__main__":
    main()
    sys.stdin = sys.__stdin__
