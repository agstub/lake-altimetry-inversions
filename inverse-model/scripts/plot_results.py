# This file contains the plot function for plotting the inversion results
# pass a list of time steps "timestep":
# if the length of timesteps is >1 then this will plot a png at each timestep (in movie directory)
# if the length of timesteps is == 1 then this will just plot a single png snapshot labeled "snap"
import sys,os
sys.path.insert(0, '../source')
from post_process import calc_dV_w,calc_dV_h
import numpy as np
import matplotlib.pyplot as plt
from params import data_dir,H,t0,x0,y0,Nt,Nx,Ny,x,y,results_dir,t,lake_name,u_d,v_d,t_r
from matplotlib.transforms import Bbox
from shapely.geometry import Point

if lake_name != 'synth' and lake_name != 'nonlinear':
    from load_lakes import gdf

def plot(t_ref,timesteps=range(Nt),h_lim=5,w_lim=5):
    if os.path.isdir(results_dir+'/movie')==False:
        os.mkdir(results_dir+'/movie')

    w_inv = np.load(results_dir+'/w_inv.npy')

    h_obs = np.load(data_dir+'/h_obs.npy')
    off_lake = np.load(data_dir+'/off_lake.npy')
    x_d = np.load(data_dir+'/x_d.npy')
    y_d = np.load(data_dir+'/y_d.npy')
    xp,yp = np.meshgrid(x_d,y_d)

    # define anomaly relative to reference time
    i0 = np.argmin(np.abs(t0-t_ref))
    h_ref = h_obs[i0,:,:] + 0*h_obs
    h_obs -= h_ref

    if lake_name != 'synth' and lake_name != 'nonlinear':
        outline = gdf.loc[gdf['name']==lake_name]
        alt_bdry = 0*xp
        for i in range(x_d.size):
            for j in range(y_d.size):
                point = Point(x_d[i],y_d[j])
                alt_bdry[j,i] = outline.contains(point)
    else:
        xc,yc = 0,0
        alt_bdry = 0*xp+1
        rad = 12
        alt_bdry[np.sqrt((xp-xc)**2+(yp-yc)**2)>rad] = 0

    if os.path.isfile(data_dir+'/w_true.npy')==True:
        w_true = np.load(data_dir+'/w_true.npy')
        dV_true = calc_dV_w(w_true,alt_bdry)
    else:
        dV_true = np.zeros(Nt)

    # calculate volume change time series:
    dV_inv = calc_dV_w(w_inv,alt_bdry)           # volume change from inversion

    dV_h = calc_dV_h(h_obs,alt_bdry)             # surface-based volume change
   
    # plot everything

    dV_max = np.max(np.array([np.max(dV_inv),np.max(dV_h)]))*(H**2)/1e9+np.sqrt(np.var(dV_inv))*(H**2)/1e9
    dV_min = np.min(np.array([np.min(dV_inv),np.min(dV_h)]))*(H**2)/1e9-np.sqrt(np.var(dV_inv))*(H**2)/1e9

    err_pct = 100*np.max(np.abs(dV_h-dV_inv))/np.max(np.abs(dV_h))
    err_vol = np.max(np.abs(dV_h-dV_inv))*(H**2)/1e9

    print('maximum discrepancy = '+str(err_vol)+' km^3')
    print('err pct = '+str(err_pct)+'\%')


    for i in timesteps:
        print('Saving image '+str(timesteps.index(i)+1)+' out of '+str(np.size(timesteps))+' \r',end='')

        fig = plt.figure(figsize=(12,12))
        plt.suptitle(r'(a) data and inversion at $t=$'+format(t0[i],'.1f')+' yr',y=1.04,x=0.4,fontsize=26,bbox=dict(boxstyle='round', facecolor='w', alpha=1))
        plt.subplot(212)
       
        if len(timesteps)>1:
            j = i
        else:
            j = -1    

        dV_invs = dV_inv[0:j]*(H**2)/1e9
        dV_alt = dV_h[0:j]*(H**2)/1e9

        if lake_name != 'synth' and lake_name != 'nonlinear':
            label = r'ICESat-2 ($\Delta V_\mathrm{alt}$)'
            dV_t = 0*dV_true[0:j]
        else:
            label = r'altimetry ($\Delta V_\mathrm{alt}$)'
            dV_t = dV_true[0:j]*(H**2)/1e9

        if j>1 or j == -1:
            lower = np.min([dV_invs[-1],dV_alt[-1] ])
            upper = np.max([dV_invs[-1],dV_alt[-1] ])
            mid = 0.5*(lower+upper)

        plt.plot(t0[0:j],dV_alt,color='indianred',linewidth=5,label=label)
        plt.plot(t0[0:j],dV_invs,color='royalblue',linewidth=4,label=r'inversion ($\Delta V_\mathrm{inv}$)')
        if j>1 or j==-1:
            err_pct = 100*np.max(np.abs(dV_alt-dV_invs))/np.max(np.abs(dV_alt))
            plt.plot([t0[-1],t0[-1]],[lower,upper],'k-',linewidth=4,clip_on=False)
            plt.plot([t0[-1]],[lower],'k-',marker='v',linewidth=5,markersize=8,clip_on=False)
            plt.plot([t0[-1]],[upper],'k-',marker='^',linewidth=5,markersize=8,clip_on=False)
            if t0[i] > t_r/3.154e7 or j==-1:
                plt.annotate(xy=(1.01*t0[-1],mid),text='{:.0f}'.format(err_pct)+'% of \n$\max\,|\Delta V_\mathrm{alt}|$',fontsize=16, annotation_clip=False, verticalalignment='center') 


        if lake_name == 'synth' or lake_name == 'nonlinear':
            plt.plot(t0[0:j],dV_t,color='k',linestyle=':',linewidth=8,label=r'true solution')


        # add viscous relaxation timescale
        plt.axvline(x=t_r/3.154e7,color='k',linestyle='--',linewidth=1,alpha=0.75)
        plt.annotate(xy=(t_r/3.154e7,-0.3),text=r'$t_\mathrm{relax}$',fontsize=14)

        plt.ylabel(r'$\Delta V$ (km$^3$)',fontsize=20)
        plt.xlim(0,t0.max())
        plt.ylim(dV_min-0.1,0.9*dV_max+0.1)
        plt.xlabel(r'$t$ (yr)',fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        if lake_name != 'synth' and lake_name != 'nonlinear':
            plt.legend(fontsize=20,ncol=2,bbox_to_anchor=(0.875,-0.15))
        else:
            plt.legend(fontsize=20,ncol=3,bbox_to_anchor=(1.04,-0.15))

        # plot off-lake component
        ax2 = plt.gca().twinx()
        dd = 0.1 # spacing for ticks
        disp = 2 # displacement so ticks appear near bottom of plot
        o_min = np.floor(np.min(off_lake)/dd)*dd
        o_max = np.ceil(np.max(off_lake)/dd)*dd
        ticks = np.arange(o_min-dd,o_max+dd,dd)
        ax2.plot(t0[0:j],off_lake[0:j],'--',color='forestgreen',linewidth=3,alpha=0.5)
        plt.yticks(ticks,fontsize=14,color='forestgreen')
        ax2.tick_params(axis='both', which='major', labelsize=14,color='forestgreen',labelcolor='forestgreen')
        ax2.set_ylim(o_min,o_max+disp)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        ax2.set_ylabel(r'$\Delta h_\mathrm{off}$ (m)',fontsize=14,loc='bottom',color='forestgreen')

        plt.title(r'(b) volume change time series', fontsize=26,bbox=dict(boxstyle='round', facecolor='w', alpha=1),y=0.91,x=0.3075)


        plt.subplot(221)
        p=plt.contourf(xp,yp,h_obs[i,:,:],cmap='PuOr_r',extend='both',levels=h_lim*np.linspace(-1,1,6))

        if lake_name != 'synth' and lake_name != 'nonlinear':
            outline.plot(edgecolor='gray',facecolor='none',ax=plt.gca(),linewidth=4)
        else:
           circle = plt.Circle((0, 0), rad, color='gray',fill=False,linewidth=4)
           plt.gca().add_patch(circle)

        speed = np.sqrt(u_d**2+v_d**2)+1e-30
        du = 5*u_d/speed
        dv = 5*v_d/speed

        if du >= 0:
            x0 = x_d.min() + 5
        else:
            x0 = x_d.min() + 7.5    
        if dv > 0 :
            y0 = y_d.max() - 7.5
        else:
            y0 = y_d.max() - 5     
       
        if speed > 1e-12:
            plt.arrow(x0,y0,du,dv,width=0.5,fc='k',ec='k',clip_on=False)
            plt.annotate(xy=(x_d.min()+6,y_d.max()-4),text=r'$\bar{\mathbf{u}}$',fontsize=20)
        else:
            plt.annotate(xy=(x_d.min()+2,y_d.max()-6),text=r'$\bar{\mathbf{u}}=\mathbf{0}$',fontsize=20)


        plt.ylabel(r'$y$ (km)',fontsize=20)
        plt.xlabel(r'$x$ (km)',fontsize=20)
        plt.xlim(x_d.min(),x_d.max())
        plt.ylim(y_d.min(),y_d.max())
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.gca().set_aspect('equal', 'box')
        cbar_ax = fig.add_axes([0.13, 0.9, 0.35, 0.02])
        cbar = fig.colorbar(p,orientation='horizontal',cax=cbar_ax)
        cbar.set_label(r'$\Delta h_\mathrm{a}$ (m)',fontsize=24,labelpad=15)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')

        plt.subplot(222)
        p=plt.contourf(xp,yp,w_inv[i,:,:],cmap='PuOr_r',levels=w_lim*np.linspace(-1,1,6),extend='both')

        if lake_name != 'synth' and lake_name != 'nonlinear':
            outline.plot(edgecolor='gray',facecolor='none',ax=plt.gca(),linewidth=4)
        else:
           circle = plt.Circle((0, 0), rad, color='gray',fill=False,linewidth=4)
           plt.gca().add_patch(circle)

        plt.xlabel(r'$x$ (km)',fontsize=20)
        plt.ylabel(r'$y$ (km)',fontsize=20)
        plt.xlim(x_d.min(),x_d.max())
        plt.ylim(y_d.min(),y_d.max())
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.gca().set_aspect('equal', 'box')
        cbar_ax = fig.add_axes([0.55, 0.9, 0.35, 0.02])
        cbar = fig.colorbar(p,orientation='horizontal',cax=cbar_ax)
        cbar.set_label(r'$w_\mathrm{b}$ (m/yr)',fontsize=24,labelpad=15)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        if len(timesteps)>1:
            plt.savefig(results_dir+'/movie/'+str(i),bbox_inches=Bbox([[0.15,-0.25],[12.5,13]]))
        elif len(timesteps)==1:    
            plt.savefig(results_dir+'/'+lake_name+'_snap',bbox_inches=Bbox([[0.15,-0.25],[12.5,13]]))
            plt.show()
        plt.close()
