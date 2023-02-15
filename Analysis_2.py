"""Setup"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def polyfunc(x, a,b,c,d,e,f):
    return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x**1 + f

def linear(x, m,c):
    return m*x + c

def exp(x, A,a):
    return A*np.exp(-x*a)

plt.rcParams.update({'font.size': 12})


"""Read"""
df_146 = pd.read_csv('./146_01.csv')
df_157 = pd.read_csv('./157_04.csv')
df_167 = pd.read_csv('./167_06.csv')
df_177 = pd.read_csv('./177_04.csv')
df_187 = pd.read_csv('./187_03.csv')

df_146.rename(columns={"Time t / s":"time","Voltage U_A1 / V":"U_A","Voltage U_B1 / V":"U_2"}, inplace=True)
df_157.rename(columns={"Time t / s":"time","Voltage U_A1 / V":"U_A","Voltage U_B1 / V":"U_2"}, inplace=True)
df_167.rename(columns={"Time t / s":"time","Voltage U_A1 / V":"U_A","Voltage U_B1 / V":"U_2"}, inplace=True)
df_177.rename(columns={"Time t / s":"time","Voltage U_A1 / V":"U_A","Voltage U_B1 / V":"U_2"}, inplace=True)
df_187.rename(columns={"Time t / s":"time","Voltage U_A1 / V":"U_A","Voltage U_B1 / V":"U_2"}, inplace=True)


"""Raw Plots"""
fig1_146 = plt.figure(figsize=(10,8))
ax = fig1_146.add_subplot(111)
ax.plot(df_146["U_2"], df_146["U_A"], 'k-', linewidth=1)
plt.xticks(np.arange(0, 4, 0.2))
plt.ylabel('U_A')
plt.xlabel('U_2')
plt.title('146')
plt.show()

fig1_157 = plt.figure(figsize=(10,8))
ax = fig1_157.add_subplot(111)
ax.plot(df_157["U_2"], df_157["U_A"], 'k-', linewidth=1)
plt.xticks(np.arange(0, 4, 0.2))
plt.ylabel('U_A')
plt.xlabel('U_2')
plt.title('157')
plt.show()

fig1_167 = plt.figure(figsize=(10,8))
ax = fig1_167.add_subplot(111)
ax.plot(df_167["U_2"], df_167["U_A"], 'k-', linewidth=1)
plt.xticks(np.arange(0, 4, 0.2))
plt.ylabel('U_A')
plt.xlabel('U_2')
plt.title('167')
plt.show()

fig1_177 = plt.figure(figsize=(10,8))
ax = fig1_177.add_subplot(111)
ax.plot(df_177["U_2"], df_177["U_A"], 'k-', linewidth=1)
plt.xticks(np.arange(0, 4, 0.2))
plt.ylabel('U_A')
plt.xlabel('U_2')
plt.title('177')
plt.show()

fig1_187 = plt.figure(figsize=(10,8))
ax = fig1_187.add_subplot(111)
ax.plot(df_187["U_2"], df_187["U_A"], 'k-', linewidth=1)
plt.xticks(np.arange(0, 4, 0.2))
plt.ylabel('U_A')
plt.xlabel('U_2')
plt.title('187')
plt.show()


"""Ranges"""
minima = pd.DataFrame()
range_146 = [(0.43,     0.71),
             (0.93,     1.26),
             (1.45,     1.78),
             (1.95,     2.31),
             (2.59,     2.71)]

range_157 = [(0.40,     0.72),
             (0.90,     1.26),
             (1.40,     1.77),
             (1.91,     2.28),
             (2.42,     2.81),
             (3.09,     3.20)]

range_167 = [(0.40,     0.67),
             (0.90,     1.22),
             (1.39,     1.73),
             (1.89,     2.25),
             (2.42,     2.77)]

range_177 = [(0.82,     1.12),
             (1.31,     1.62),
             (1.78,     2.15),
             (2.27,     2.65),
             (2.80,     3.14)]

range_187 = [(0.79,     1.12),
             (1.29,     1.62),
             (1.76,     2.13),
             (2.25,     2.64),
             (2.78,     3.11)]


"""Clean"""
#157 at 3.19, 1
df_157.drop(index=52,inplace=True)
#167 at 1.005, 1
#167 at 2.17, 6
#167 at 2.725, 1
df_167.drop(index=[49,248,479,480,481,482,483,484,485,811],inplace=True)
#187 at 0.45, 1
#187 at 0.78, 3
#187 at 1.52, 1
#187 at 1.6, 1
#187 at 1.84, 1
df_187.drop(index=[100,232,377,413],inplace=True)


"""Minima Plots"""
i = 0
df_146_min = pd.DataFrame({'min':[],'n':[]})
for low,up in range_146:
    df_min = df_146.loc[df_146["U_2"].between(low,up)]
    
    fig2_146 = plt.figure(figsize=(10,8))
    ax = fig2_146.add_subplot(111)
    
    ax.plot(df_min["U_2"], df_min["U_A"], "xk", lw=1)
    ax.set_xlabel("U_2")
    ax.set_ylabel("U_A")
    
    popt, pcov = curve_fit(polyfunc, df_min["U_2"], df_min["U_A"])
    linefit_time = np.linspace(df_min["U_2"].min(), df_min["U_2"].max(), 40)
    ax.plot(linefit_time, polyfunc(linefit_time, *popt), "k-", lw=1)
    
    linefit = pd.DataFrame(data={"U_2":linefit_time,"U_A":polyfunc(linefit_time, *popt)})
    df = pd.DataFrame({'min':[min(linefit["U_2"])],'n':[i]})
    df_146_min = df_146_min.append(df, ignore_index=True)
    i = i + 1


i = 0
df_157_min = pd.DataFrame({'min':[],'n':[]})
for low,up in range_157:
    df_min = df_157.loc[df_157["U_2"].between(low,up)]
    
    fig2_157 = plt.figure(figsize=(10,8))
    ax = fig2_157.add_subplot(111)
    
    ax.plot(df_min["U_2"], df_min["U_A"], "xk", lw=1)
    ax.set_xlabel("U_2")
    ax.set_ylabel("U_A")
    
    popt, pcov = curve_fit(polyfunc, df_min["U_2"], df_min["U_A"])
    linefit_time = np.linspace(df_min["U_2"].min(), df_min["U_2"].max(), 40)
    ax.plot(linefit_time, polyfunc(linefit_time, *popt), "k-", lw=1)
    
    linefit = pd.DataFrame(data={"U_2":linefit_time,"U_A":polyfunc(linefit_time, *popt)})
    df = pd.DataFrame({'min':[min(linefit["U_2"])],'n':[i]})
    df_157_min = df_146_min.append(df, ignore_index=True)
    i = i + 1


i = 0
df_167_min = pd.DataFrame({'min':[],'n':[]})
for low,up in range_167:
    df_min = df_167.loc[df_167["U_2"].between(low,up)]
    
    fig2_167 = plt.figure(figsize=(10,8))
    ax = fig2_167.add_subplot(111)
    
    ax.plot(df_min["U_2"], df_min["U_A"], "xk", lw=1)
    ax.set_xlabel("U_2")
    ax.set_ylabel("U_A")
    
    popt, pcov = curve_fit(polyfunc, df_min["U_2"], df_min["U_A"])
    linefit_time = np.linspace(df_min["U_2"].min(), df_min["U_2"].max(), 40)
    ax.plot(linefit_time, polyfunc(linefit_time, *popt), "k-", lw=1)
    
    linefit = pd.DataFrame(data={"U_2":linefit_time,"U_A":polyfunc(linefit_time, *popt)})
    df = pd.DataFrame({'min':[min(linefit["U_2"])],'n':[i]})
    df_167_min = df_167_min.append(df, ignore_index=True)
    i = i + 1


i = 0
df_177_min = pd.DataFrame({'min':[],'n':[]})
for low,up in range_177:
    df_min = df_177.loc[df_177["U_2"].between(low,up)]
    
    fig2_177 = plt.figure(figsize=(10,8))
    ax = fig2_177.add_subplot(111)
    
    ax.plot(df_min["U_2"], df_min["U_A"], "xk", lw=1)
    ax.set_xlabel("U_2")
    ax.set_ylabel("U_A")
    
    popt, pcov = curve_fit(polyfunc, df_min["U_2"], df_min["U_A"])
    linefit_time = np.linspace(df_min["U_2"].min(), df_min["U_2"].max(), 40)
    ax.plot(linefit_time, polyfunc(linefit_time, *popt), "k-", lw=1)
    
    linefit = pd.DataFrame(data={"U_2":linefit_time,"U_A":polyfunc(linefit_time, *popt)})
    df = pd.DataFrame({'min':[min(linefit["U_2"])],'n':[i]})
    df_177_min = df_177_min.append(df, ignore_index=True)
    i = i + 1


i = 0
df_187_min = pd.DataFrame({'min':[],'n':[]})
for low,up in range_187:
    df_min = df_187.loc[df_187["U_2"].between(low,up)]
    
    fig2_187 = plt.figure(figsize=(10,8))
    ax = fig2_187.add_subplot(111)
    
    ax.plot(df_min["U_2"], df_min["U_A"], "xk", lw=1)
    ax.set_xlabel("U_2")
    ax.set_ylabel("U_A")
    
    popt, pcov = curve_fit(polyfunc, df_min["U_2"], df_min["U_A"])
    linefit_time = np.linspace(df_min["U_2"].min(), df_min["U_2"].max(), 40)
    ax.plot(linefit_time, polyfunc(linefit_time, *popt), "k-", lw=1)
    
    linefit = pd.DataFrame(data={"U_2":linefit_time,"U_A":polyfunc(linefit_time, *popt)})
    df = pd.DataFrame({'min':[min(linefit["U_2"])],'n':[i]})
    df_187_min = df_187_min.append(df, ignore_index=True)
    i = i + 1


"""Linear plots"""
fig3 = plt.figure(figsize=(10,8))
ax = fig3.add_subplot(111)


ax.plot(df_146_min["n"], df_146_min["min"], "xk", lw=1)
ax.set_xlabel("n")
ax.set_ylabel("min")

popt, pcov = curve_fit(linear, df_146_min["n"], df_146_min["min"])
linefit_time = np.linspace(df_146_min["n"].min(), df_146_min["n"].max(), 40)
ax.plot(linefit_time, linear(linefit_time, *popt), "k-", lw=1)
print(popt[0])


ax.plot(df_157_min["n"], df_157_min["min"], "xk", lw=1)
ax.set_xlabel("n")
ax.set_ylabel("min")

popt, pcov = curve_fit(linear, df_157_min["n"], df_157_min["min"])
linefit_time = np.linspace(df_157_min["n"].min(), df_157_min["n"].max(), 40)
ax.plot(linefit_time, linear(linefit_time, *popt), "k-", lw=1)
print(popt[0])


ax.plot(df_167_min["n"], df_167_min["min"], "xk", lw=1)
ax.set_xlabel("n")
ax.set_ylabel("min")

popt, pcov = curve_fit(linear, df_167_min["n"], df_167_min["min"])
linefit_time = np.linspace(df_167_min["n"].min(), df_167_min["n"].max(), 40)
ax.plot(linefit_time, linear(linefit_time, *popt), "k-", lw=1)
print(popt[0])


ax.plot(df_177_min["n"], df_177_min["min"], "xk", lw=1)
ax.set_xlabel("n")
ax.set_ylabel("min")

popt, pcov = curve_fit(linear, df_177_min["n"], df_177_min["min"])
linefit_time = np.linspace(df_177_min["n"].min(), df_177_min["n"].max(), 40)
ax.plot(linefit_time, linear(linefit_time, *popt), "k-", lw=1)
print(popt[0])


ax.plot(df_187_min["n"], df_187_min["min"], "xk", lw=1)
ax.set_xlabel("n")
ax.set_ylabel("min")

popt, pcov = curve_fit(linear, df_187_min["n"], df_187_min["min"])
linefit_time = np.linspace(df_187_min["n"].min(), df_187_min["n"].max(), 40)
ax.plot(linefit_time, linear(linefit_time, *popt), "k-", lw=1)
print(popt[0])


plt.show()



"""Time of Flight"""
k_B = 1.38*10**-23 #m**2 kg s**-2 K**-1
sigma = 2.1*10**-19 #m**2
T = [140,145,150,155,160,165,170,175,180,185,190] #deg C
lamda = []
for t in T:
    p = 8.7*10**(9-(3110/(t+273.15)))
    lamda.append((k_B*t)/(p*sigma))

df_ToF = pd.DataFrame({'Temp':T, 'ToF':lamda})


fig4 = plt.figure(figsize=(10,8))
ax = fig4.add_subplot(111)

ax.plot(df_ToF["Temp"], df_ToF["ToF"], "xk", lw=1)
ax.set_xlabel("Temp")
ax.set_ylabel("Time of Flight")

p0 = [1e-9,1e-3]

linefit_time = np.linspace(df_ToF["Temp"].min(), df_ToF["Temp"].max(), 100)

popt, pcov = curve_fit(exp, df_ToF["Temp"], df_ToF["ToF"], p0=p0)
ax.plot(linefit_time, exp(linefit_time, *popt), "k-", lw=1)

plt.show()
