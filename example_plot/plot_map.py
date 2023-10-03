import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import polplot # https://github.com/klaundal/polplot
import dipole # https://github.com/klaundal/dipole
import datetime as dt

left_lt = 20
right_lt = 4
minlat = 55
fig, axes = plt.subplots(nrows = 2, figsize = (9, 10))

time = dt.datetime(2023, 10, 3, 23, 00)
dp = dipole.Dipole(epoch = time.year)

paxes = [polplot.Polarplot(ax, minlat = minlat, sector = str(left_lt) + '-' + str(right_lt)) for ax in axes.flatten()]
paxFAC, paxB = paxes

IMAGE = pd.read_table("../IMAGE_magnetometers.txt", sep=" ", skipinitialspace=True, header=None, names=["number", "code", "name", "glat", "glon", "mlat", "mlon", "provider", "network", "start_date", "end_date", ], index_col=0)
IMAGE['start_date'] = pd.to_datetime(IMAGE['start_date'])
IMAGE['end_date'  ] = pd.to_datetime(IMAGE['end_date'  ])
IMAGE['end_date'] = IMAGE.end_date.fillna(dt.datetime(2055, 1, 1)) # fill in end date if not available
IMAGE = IMAGE[(IMAGE['start_date'] < time) & (IMAGE['end_date'] > time)]

data = np.load('dB_paper_plotting.npy',  allow_pickle=True).item()
mlat = data['mlat']
mlon = data['mlon']

Be, Bn, Bu = data['Be_df'], data['Bn_df'], data['Bu_df']

glat, glon, Beg, Bng = dp.mag2geo(mlat.flatten(), mlon.flatten(), Be.flatten(), Bn.flatten())
x_, y_ = paxB._latlt2xy(glat.reshape(Be.shape), glon.reshape(Be.shape)/15, ignore_plot_limits = False)
iii = ~np.isfinite(x_)
x_, y_ = paxB._latlt2xy(glat.reshape(Be.shape), glon.reshape(Be.shape)/15, ignore_plot_limits = True)
Bu[iii] = np.nan # filter facs where coordinates are not defined

paxB.contourf(glat, glon/15, Bu, cmap = plt.cm.bwr, levels = np.linspace(-150e-9, 150e-9, 22))
paxB.quiver(glat.reshape(Be.shape)[::2,::3].flatten(), glon.reshape(Be.shape)[::2,::3].flatten()/15, Bng.reshape(Be.shape)[::2,::3].flatten(), Beg.reshape(Be.shape)[::2,::3].flatten())

facdata = np.load('fac_paper_plotting.npy', allow_pickle=True).item()
facmlat, facmlon = facdata['mlat'], facdata['mlon']
fac = facdata['fac']
facglat, facglon, _, _ = dp.mag2geo(facmlat.flatten(), facmlon.flatten(), fac.flatten(), fac.flatten())


x_, y_ = paxFAC._latlt2xy(facglat.reshape(fac.shape), facglon.reshape(fac.shape)/15, ignore_plot_limits = False)
iii = ~np.isfinite(x_)
x_, y_ = paxFAC._latlt2xy(facglat.reshape(fac.shape), facglon.reshape(fac.shape)/15, ignore_plot_limits = True)
fac[iii] = np.nan # filter facs where coordinates are not defined
paxFAC.ax.pcolormesh(x_, y_, fac, cmap = plt.cm.bwr)#.flatten(), cmap = plt.cm.bwr, levels = np.linspace(-3e-6, 3e-6, 22))



# makeup
for pax in paxes:
    pax.coastlines(linewidth = .4, color = 'grey')

    # plot dipole latitude circles
    for lat in np.r_[50:81:10]:
        lon = np.linspace(0, 360, 360)
        glat, glon, _, _ = dp.mag2geo(lat, lon, lat, lat)
        pax.plot(glat, glon/15, color = 'C0', zorder = 1, linewidth = .4)

    # plot dipole meridians
    for lon in np.r_[0:351:30]:
        lat = np.linspace(0, 90, 190)
        glat, glon, _, _ = dp.mag2geo(lat, lon, lat, lat)
        pax.plot(glat, glon/15, color = 'C0', zorder = 1, linewidth = .4)

    pax.scatter(IMAGE.glat.values,  IMAGE.glon.values/15, s = 4)

    # draw frame
    pax.plot([minlat, 90], [left_lt, left_lt], color = 'black')
    pax.plot([minlat, 90], [right_lt, right_lt], color = 'black')
    pax.plot( np.full(100, minlat), np.linspace(left_lt, 24 + right_lt, 100), color = 'black')

plt.tight_layout()
plt.show()

