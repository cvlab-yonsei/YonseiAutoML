import torch

def get_struc():
    struc = []
    base = torch.zeros(6,5)
    for i in range(5):   
        base[0,i] = 1
        
        for ii in range(5):
            base[1,ii] = 1     
            
            for iii in range(5):
                base[2,iii]=1
                
                for j in range(5):
                    base[3,j] = 1
                    
                    for jj in range(5):
                        base[4,jj] = 1
                        
                        for jjj in range(5):
                            base[5,jjj] = 1
                            
                            struc.append(base.clone())
                        
                            
                            base[5] = 0
                        base[4] = 0
                    base[3] = 0
                base[2] = 0
            base[1] = 0
        base[0] = 0

    return struc