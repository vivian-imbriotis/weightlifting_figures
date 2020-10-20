import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rpy2 import robjects

sns.set_style("darkgrid")

#These are the coefficients in the Wilk's Score polynomial
wilks_values_men = np.array((-216.0475144,16.2606339,
                             -0.002388645,-0.00113732,
                             7.01863E-06,-1.291e-08))

wilks_values_women = np.array((594.31747775582, -27.23842536447,
                               0.82112226871,-0.00930733913,
                               4.731582E-05,-9.054E-08))

lifting_goals = dict(zip(("squat","bench","deadlift","press","row"),
                         (120,75,150,50,75)))

class LinearRegression():
    def __init__(self):
        '''
        An object for performing Linear Regression using the
        R lm(...) interface.

        Returns
        -------
        Linear Regression object.

        '''
        self.model = None
    def fit(self,X,Y):
        robjects.globalenv["X"] = robjects.FloatVector(X)
        robjects.globalenv["Y"] = robjects.FloatVector(Y)
        self.model = robjects.r("lm(Y~X)")
        return self
    def predict(self,X):
        robjects.globalenv["model"] = self.model
        robjects.globalenv["X"] = robjects.FloatVector(X)
        result = robjects.r("predict(model,data.frame(X))")
        return np.array(result)
    def intervals(self,X,interval=0.8):
        robjects.globalenv["model"] = self.model
        robjects.globalenv["X"] = robjects.FloatVector(X)
        result = robjects.r(f"predict(model,data.frame(X),interval='prediction',level={interval})")
        result = np.array(result)
        return result.transpose()

class LogarithmicRegression(LinearRegression):
        '''
        An object for performing Logarithmic Regression using the
        R lm(...) interface.

        Returns
        -------
        Logarithmic Regression object.

        '''
    def __init__(self,asymptote=3):
        self.asymptote = asymptote
        super().__init__()
    def fit(self,X,Y):
        robjects.globalenv["X"] = robjects.FloatVector(X)
        robjects.globalenv["Y"] = robjects.FloatVector(Y)
        self.model = robjects.r(f"lm(Y~I(log(X + {self.asymptote})))")
        return self
    
    
            

def calc_1rm(weight,reps):
    '''
    Calculate a one-rep max from an as-many-reps-as-possible set using the
    Brzycki method.

    Parameters
    ----------
    weight : float
        Amount of weight lifted (kg).
    reps : int
        Number of repetitions.

    Returns
    -------
    int
        Projected one-repetition maximum.

    '''
    if pd.isnull(weight) or pd.isnull(reps): return np.nan
    raw_1rm = weight/(1.0278 - 0.0278*reps)
    return 2.5*round(raw_1rm/2.5)


def wilks_coef(bodyweight, gender='male'):
    '''
    Calculate a lifter's Wilk's coefficient based on bodyweight. This value
    is used to compare lifts between lifters of different bodyweights.

    Parameters
    ----------
    bodyweight : float
        Bodyweight (kg).
    gender : str, optional
        Whether to use the male or female version of the Wilk's formula.
        Must be 'male' or 'female'. The default is 'male' (because I am male).

    Raises
    ------
    ValueError
        Raised for innappropriate values of gender kwarg.

    Returns
    -------
    float
        Wilks coefficient.

    '''
    w = bodyweight
    bodyweight_polynomial = np.array((1,w,w**2,w**3,w**4,w**5))
    if gender in ("man","male"):
        values = wilks_values_men
    elif gender in ("woman","female"):
        values = wilks_values_women
    else: raise ValueError("Gender must be 'male' or 'female' for the purpose"
                           " of Wilk's Scoring")
    return 500 / np.sum(values*bodyweight_polynomial)


def wilks_score(weight, bodyweight, gender='male'):
    '''
    Compute a Wilk's score for a lift.

    Parameters
    ----------
    weight : float
        Amount of weight lifted (kg)
    bodyweight : float
        Lifter bodyweight (kg).
    gender : str, optional
        Which polynomial to use, 'female' or 'male'. The default is 'male' 
        (because I am male).

    Returns
    -------
    float
        The wilks score.

    '''
    return weight*wilks_coef(bodyweight,gender)

def get_1rms_from_csv(path):
    '''
    Read in data from a formatted CSV and get a dataframe of approximate
    one-repetition max lifts.

    Parameters
    ----------
    path : str
        path to csv.

    Returns
    -------
    df : pandas.DataFrame
        A pandas dataframe of one-repetition maxes, with colums ("squat",
        "bench","deadlift","press","row").

    '''
    df = pd.read_csv(path)
    squat = [calc_1rm(weight, reps) for weight, reps in zip(df["Squat Weight"],df["s Reps"])]
    bench = [calc_1rm(weight, reps) for weight, reps in zip(df["Bench Weight"],df["b Reps"])]
    deadl = [calc_1rm(weight, reps) for weight, reps in zip(df["Deadlift Weight"],df["d Reps"])]
    press = [calc_1rm(weight, reps) for weight, reps in zip(df["OHP weight"],df["p Reps"])]
    row   = [calc_1rm(weight, reps) for weight, reps in zip(df["Row weight"],df["r reps"])]
    df = pd.DataFrame(np.array((squat,bench,deadl,press,row)).transpose(),
                      columns = ("squat","bench","deadlift","press","row"))
    return df

def get_weight_from_excel(path):
    '''
    Read in data from TDEE 3.0 excel datasheet and get bodyweight
    per day

    Parameters
    ----------
    path : str
        path to excel file.

    Returns
    -------
    np.array
        bodyweight indexed by dat.

    '''
    df = pd.read_excel(path)
    return df.iloc[range(10,16*2,2),3:10].values.flatten()


def create_1rm_figure(weight,one_rep_maxes):
    '''
    Create a figure showing one-repetition maxes and bodyweight
    with OLS forecasting.

    Parameters
    ----------
    weight : array
        array read in via get_weight_from_excel.
    one_rep_maxes : dataframe
        dataframe read in via get_1rms_from_csv.

    Returns
    -------
    fig : matplotlib.Figure
        The created figure.

    '''
    fig,ax = plt.subplots(nrows = 2, ncols = 3, tight_layout=True,
                          figsize = [12,8])
    for lift, axis in zip(one_rep_maxes.columns,ax.flatten()[:-1]):
        axis.set_title(lift.capitalize())
        axis.plot(one_rep_maxes[lift],marker='o',label="1-rep max (Brzycki method)")
        axis.hlines(lifting_goals[lift],0,16,linestyle="--",label="Target weight")
        one_rms = one_rep_maxes[lift]
        one_rms = one_rms[~pd.isnull(one_rms)].values
        model = LogarithmicRegression().fit(np.arange(0,len(one_rms)).reshape(-1,1),
                            one_rms)
        forecast,lwr,upr = model.intervals(np.arange(0,17).reshape(-1,1))
        # axis.plot(forecast,linestyle = "--", label = "Prediction")
        axis.fill_between(np.arange(0,17),lwr,upr,color=sns.color_palette()[3],
                           alpha = 0.4,label = "80% Prediction Interval (Log OLS)")
        ymin,ymax = axis.get_ylim()
        axis.set_ylim((ymin,lifting_goals[lift]+
            (lifting_goals[lift] - ymin)*0.1))
    weight_ax = ax.flatten()[-1]
    weight_ax.set_title("Bodyweight")
    weight_ax.plot(np.linspace(-1,16,len(weight)),weight)
    weight_ax.hlines(75,-1,16,linestyle="--",label="Target weight")
    model = LinearRegression().fit(np.linspace(0,16,len(weight))[~pd.isnull(weight)].reshape(-1,1),
                        weight[~pd.isnull(weight)])
    ymin,ymax = weight_ax.get_ylim()
    weight_ax.set_ylim((ymin,75+(75 - ymin)*0.1))
    forecast,lwr,upr = model.intervals(np.arange(0,17).reshape(-1,1))
    # weight_ax.plot(forecast,linestyle = "--", label = "Prediction")
    weight_ax.fill_between(np.arange(0,17),lwr,upr,color=sns.color_palette()[4],
                           alpha = 0.4, label = "80% prediction interval (Linear OLS)")
    weight_ax.legend(loc="lower right")
    ax[0][0].legend(loc='lower right')
    for axis in ax[:,0]: axis.set_ylabel("Mass (kg)")
    for axis in ax[-1,:]:axis.set_xlabel("Weeks into program")
    for axis in ax[:-1,:].flatten(): axis.set_xticklabels([])
    return fig

def create_wilks_figure(weight,one_rms):
    '''
    Create a figure showing wilk's score progress
    with OLS forecasting.

    Parameters
    ----------
    weight : array
        array read in via get_weight_from_excel.
    one_rep_maxes : dataframe
        dataframe read in via get_1rms_from_csv.

    Returns
    -------
    fig : matplotlib.Figure
        The created figure.

    '''
    wilks = np.full(17,np.nan)
    for week,_ in enumerate(wilks):
        w = weight[week*7:(week+1)*7].astype(float)
        mean_weight = np.nanmean(w)
        total_lifts = np.sum((one_rms.bench[week],
                                 one_rms.squat[week],
                                 one_rms.deadlift[week]))
        wilks[week] = wilks_score(total_lifts,mean_weight)
    fig,ax = plt.subplots()
    ax.plot(wilks,marker = 'o',label="Overall Wilk's Score (incorperating bench, press, DL and squat)")
    ax.set_xlabel("Weeks into program")
    ax.set_ylabel("Overall Wilks Score")
    ax.set_xlim(-1,17)
    ax.set_ylim((0,300))
    model = LogarithmicRegression().fit(np.arange(0,17)[~pd.isnull(wilks)].reshape(-1,1),
                        wilks[~pd.isnull(wilks)])
    forecast,lwr,upr = model.intervals(np.arange(0,17).reshape(-1,1))
    ax.fill_between(np.arange(0,17),lwr,upr,color=sns.color_palette()[3],
                           alpha = 0.4,label = "80% Prediction Interval (Log OLS)")
    ax.legend()
    return fig

if __name__=="__main__":
    plt.close('all')
    weight = get_weight_from_excel("TDEE.xlsx")
    one_rep_maxes = get_1rms_from_csv("weight progress.csv")
    fig = create_1rm_figure(weight,one_rep_maxes)
    fig.show()
    fig2 = create_wilks_figure(weight,one_rep_maxes)
    fig2.show()