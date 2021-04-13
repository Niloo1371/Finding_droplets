import numpy as np
import random


class Series(object):
    _counter = 0  # Number of Series we have

    def __init__(self,
                 date_string=None,
                 T_function=None,
                 d_value=None,
                 d_sigma=None,
                 s_value=None,
                 s_sigma=None,
                 q_value=None):

        Series._counter += 1  # increase the count by one
        self.id = Series._counter  # Id of this series

        self.date = date_string
        # self.measured_points=[]
        # self.calculated_points=[]
        self.descriptor = date_string
        self.T_function = T_function

        # descriptors for drops
        self.d_value = d_value
        self.d_sigma = d_sigma
        self.s_value = s_value
        self.s_sigma = s_sigma
        self.q_value = q_value

    def calculate_t(self, distance, d, q, s):
        '''
        calculate_t(distance, d, q, s)
        use 

        l in m
        d in m
        q in microliters/s
        s in m
        '''

        d_um = d*1e6
        # l_mm=l*1e3
        q_m3_per_s = q*1.666666e-11
        # s_um=s*1e6

        #print(str(l) + ' m distance')
        #print(str(l_mm) + ' mm distance')
        #print(str(d) + ' m diameter')
        #print(str(d_um) + ' um diameter')
        #print(str(s) + ' m spacing')
        #print(str(s_um) + ' um spacing')
        #print(str(q_m3_per_s) + ' flow rate in m^3/s')

        t = np.divide(distance*np.pi*d*d*d, 6*q_m3_per_s*s)
        return t

    def calculate_v(self, d, q, s):
        '''
        calculate_v(d, q, s)
        use 

        d in m
        q in microliters/s
        s in m
        '''

        d_um = d*1e6
        # l_mm=l*1e3
        q_m3_per_s = q*1.666666e-11
        # s_um=s*1e6

        #print(str(d) + ' m diameter')
        #print(str(d_um) + ' um diameter')
        #print(str(s) + ' m spacing')
        #print(str(s_um) + ' um spacing')
        #print(str(q_m3_per_s) + ' flow rate in m^3/s')

        v = np.divide(6*q_m3_per_s*s, np.pi*d*d*d)
        return v

    def calculate_T(self, t, d):
        '''
        input is time and diameter, output is temperature. 

        please use 

        d in m
        t in s
        T_function is the interpolation function to use
        '''
        try:
            T = self.T_function([d*1e6, t], method='linear')[0]
        except:
            #print('{0} {1}'.format(t,d*1e6))
            T = np.nan
        return T
    
    def calculate_volume(self, d):
        '''
        input is observables, output is V in unit of m^3. 

        d in m
        '''
        r=d/2
        V = np.multiply(4.0/3.0, np.power(r * np.pi, 3))

        return V
    
    def calculate_J(self, q, s, d, l_1, l_2, f_1, f_2):
        '''
        input is observables, output is J in unit of 1/(s m^3). 

        please use 

        q in microliters/s
        s in m
        d in m
        l in m
        f as fraction

        we will calculate at length l_1 with fraction f_2
        '''

        q_m3_per_s = q*1.666666e-11

        the_log = np.log(np.divide(np.subtract(1, f_2), np.subtract(1, f_1)))
        factor = np.divide((-36*np.multiply(q_m3_per_s, s)),
                           (np.multiply(np.power(np.pi, 2), np.power(d, 6))))

        J = np.divide(np.multiply(factor, the_log), np.subtract(l_2, l_1))

        return J

    def calculate_J_method2(self, q, s, d, l_1, l_2, f_1, f_2):
        '''
        METHOD 2! 
        From 

        https://doi.org/10.1021/acs.jpca.6b03843

        Atkinson 2016


        input is observables, output is J in unit of 1/(s m^3). 

        please use 

        q in microliters/s
        s in m
        d in m
        l in m
        f as fraction

        we will calculate at length l_1 with fraction f_2
        '''

        q_m3_per_s = q*1.666666e-11

        the_log = np.log(np.subtract(np.subtract(1, f_2), np.subtract(1, f_1)))
        factor = np.divide((-36*np.multiply(q_m3_per_s, s)),
                           (np.multiply(np.power(np.pi, 2), np.power(d, 6))))

        J = np.divide(np.multiply(factor, the_log), np.subtract(l_2, l_1))

        return J

    def testing(self):
        print('testing runs')

    def set_observables(self,
                        frequency=None,
                        d_value=None,
                        d_sigma=None,
                        s_value=None,
                        s_sigma=None,
                        q_value=None,
                        distances=None,
                        counts=None,
                        fractions=None,
                        type_counts=['korv']):
        '''
        type_counts is an array of the different types with length the same as distances:
        [[12,13,44],[12,13,44],[12,13,44],[12,13,44]]
        '''

        assert len(set(map(len, (distances, counts, fractions)))
                   ) == 1, "Lists must have equal length"

        if type_counts[0] != 'korv':
            # If classes of droplets are input
            self.number_of_types = len(type_counts[0])
            #print(type_counts)


            #self.type_fractions = np.divide(np.array(type_counts).T, counts)
            self.type_fractions = np.array([np.divide(type_counts[total_count_index],counts[total_count_index]) for total_count_index,total_count in enumerate(counts)])
            
            #print(self.type_fractions)
            self.do_types = True
        else:
            self.do_types = False

        self.d_value = d_value
        self.d_sigma = d_sigma
        self.s_value = s_value
        self.s_sigma = s_sigma
        self.q_value = q_value
        self.frequency = frequency

        self.ideal_t_at_measured = []
        self.ideal_t_at_calculated = []
        self.measured_distances = []

        self.ideal_v = self.calculate_v(d_value, q_value, s_value)

        for distance_i, distance in enumerate(distances):
            # set measured distances
            self.measured_distances.append(distance)
            # calculate t at those
            self.ideal_t_at_measured.append(self.calculate_t(
                distance, self.d_value, self.q_value, self.s_value))

            self.counts = counts
            self.fractions = fractions

        # calculate mid-point distances and calculate t at those
        calculated_distances = np.convolve(distances, [1, 1], 'valid') / 2
        self.calculated_distances = calculated_distances
        self.ideal_t_at_calculated = [self.calculate_t(
            distance, self.d_value, self.q_value, self.s_value) for distance in calculated_distances]

        # calculate ideal T:

        # print(self.d_value)
        # print(self.ideal_t_at_measured)
        self.ideal_T_at_measured = [self.calculate_T(
            t, self.d_value) for t in self.ideal_t_at_measured]

        # print(self.ideal_t_at_calculated)

        self.ideal_T_at_calculated = [self.calculate_T(
            t, self.d_value) for t in self.ideal_t_at_calculated]

        # calculate ideal J:
        self.ideal_J = [self.calculate_J(self.q_value,
                                         self.s_value,
                                         self.d_value,
                                         self.measured_distances[i+1],
                                         self.measured_distances[i],
                                         self.fractions[i+1],
                                         self.fractions[i]) for i in range(len(self.calculated_distances))]

    def calculate_errors(self, percentile=5, trials=1000, do_J=False):
        # for the matrix varibles:

        self.trials = trials

        assert percentile > 0 and percentile <= 50, "percentile must be >0 and <=50"
        self.percentiles = (percentile, 100-percentile)

        self.run_MC_for_only_T(trials)

        # high and low limits
        self.low_T_at_measured = np.percentile(
            self.T_MC_matrix_at_measured, q=percentile, axis=0)
        self.high_T_at_measured = np.percentile(
            self.T_MC_matrix_at_measured, q=100-percentile, axis=0)
        # errors for plotting
        self.low_T_err_at_measured = np.subtract(
            self.ideal_T_at_measured, self.low_T_at_measured)
        self.high_T_err_at_measured = np.subtract(
            self.high_T_at_measured, self.ideal_T_at_measured)

        # high and low limits
        self.low_t_at_measured = np.percentile(
            self.t_MC_matrix_at_measured, q=percentile, axis=0)
        self.high_t_at_measured = np.percentile(
            self.t_MC_matrix_at_measured, q=100-percentile, axis=0)
        # errors for plotting
        self.low_t_err_at_measured = np.subtract(
            self.ideal_t_at_measured, self.low_t_at_measured)
        self.high_t_err_at_measured = np.subtract(
            self.high_t_at_measured, self.ideal_t_at_measured)

        # high and low limits
        self.low_v_at_measured = np.percentile(
            self.v_MC_matrix_at_measured, q=percentile, axis=0)
        self.high_v_at_measured = np.percentile(
            self.v_MC_matrix_at_measured, q=100-percentile, axis=0)
        # errors for plotting
        self.low_v_err_at_measured = np.subtract(
            self.ideal_v, self.low_v_at_measured)
        self.high_v_err_at_measured = np.subtract(
            self.high_v_at_measured, self.ideal_v)

        if do_J:
            self.run_MC_for_J(trials)

            # high and low limits
            self.low_J = np.percentile(self.J_MC_matrix, q=percentile, axis=0)
            self.high_J = np.percentile(
                self.J_MC_matrix, q=100-percentile, axis=0)
            # errors for plotting
            self.low_J_err = np.subtract(self.ideal_J, self.low_J)
            self.high_J_err = np.subtract(self.high_J, self.ideal_J)

            # high and low limits
            self.low_T = np.percentile(self.T_MC_matrix, q=percentile, axis=0)
            self.high_T = np.percentile(
                self.T_MC_matrix, q=100-percentile, axis=0)
            # errors for plotting
            self.low_T_err = np.subtract(
                self.ideal_T_at_calculated, self.low_T)
            self.high_T_err = np.subtract(
                self.high_T, self.ideal_T_at_calculated)

            # high and low limits
            self.low_t = np.percentile(self.t_MC_matrix, q=percentile, axis=0)
            self.high_t = np.percentile(
                self.t_MC_matrix, q=100-percentile, axis=0)
            # errors for plotting
            self.low_t_err = np.subtract(
                self.ideal_t_at_calculated, self.low_t)
            self.high_t_err = np.subtract(
                self.high_t, self.ideal_t_at_calculated)

            # high and low limits
            self.low_v = np.percentile(self.v_MC_matrix, q=percentile, axis=0)
            self.high_v = np.percentile(
                self.v_MC_matrix, q=100-percentile, axis=0)
            # errors for plotting
            self.low_v_err = np.subtract(self.ideal_v, self.low_v)
            self.high_v_err = np.subtract(self.high_v, self.ideal_v)

    def randomize_droplet_type(self, observed_fractions_array, total_counts):
        '''
        input:
        observed_fractions_array  : array of fraction of all types including water 
        total_counts              : total number fo droplets

        output:
        randomized_fraction_array :randomized fractions 
        assigned_counts           :radnomized counts


        Example:

        We have 12 droplets, of these 6 are type1, 3 are type2. This means 
        fraction_water=3/12=0.25
        fraction_type1=6/12=0.5
        fractiom_type2=3/12=0.25


        In one sampling, we can do:
        randomize_droplet_class([fraction_water,fraction_type1,fraction_type2],12)

        output coould then be 
        (array([0.16666667, 0.75      , 0.08333333]), array([2., 9., 1.]))
        where the first array is the randomized fractions, and the second array is the randomized counts
        '''

        number_of_fractions = len(observed_fractions_array)
        index_for_fractions = np.arange(number_of_fractions)

        bins = np.arange(number_of_fractions+1)

        assigned_counts = np.zeros(number_of_fractions)

        for count_index in range(total_counts):
            type_array, _ = np.histogram(random.choices(
                index_for_fractions, observed_fractions_array), bins=bins)
            assigned_counts = np.add(assigned_counts, type_array)

        randomized_fraction_array = np.divide(assigned_counts, total_counts)

        return randomized_fraction_array, assigned_counts

    def run_MC_for_only_T(self, trials=1000):
        '''        
        In each trial, A value is sampled for q,d and s

        t and T errors can then be calculated for all points

        Then arrays for t and T are calculated

        '''

        self.number_of_trials_only_T = trials

        len_calculated = len(self.measured_distances)

        self.t_MC_matrix_at_measured = np.zeros([trials, len_calculated])
        self.T_MC_matrix_at_measured = np.zeros([trials, len_calculated])
        self.v_MC_matrix_at_measured = np.zeros([trials, len_calculated])
        self.volume_MC_array_at_measured = np.zeros(trials)
        self.fractions_MC_matrix_at_measured = np.zeros(
            [trials, len_calculated])

        self.distances_matrix_at_measured = np.zeros([trials, len_calculated])

        if self.do_types:
            self.type_count_MC_matrix_at_measured = np.empty(
                [trials, len_calculated,self.number_of_types])
            self.type_fraction_MC_matrix_at_measured = np.empty(
                [trials, len_calculated,self.number_of_types])

        for trial in range(trials):

            trial_q = self.q_value
            trial_d = np.random.normal(loc=self.d_value, scale=self.d_sigma)
            trial_volume = self.calculate_volume(trial_d)
            trial_s = np.random.normal(loc=self.s_value, scale=self.s_sigma)

            trial_t_array = [self.calculate_t(self.measured_distances[i],
                                              trial_d,
                                              trial_q,
                                              trial_s) for i in range(len_calculated)]

            trial_v_array = [self.calculate_v(
                trial_d, trial_q, trial_s) for i in range(len_calculated)]

            trial_f_array = [np.divide(np.random.binomial(
                self.counts[i], self.fractions[i]), self.counts[i]) for i in range(len(self.counts))]

            if self.do_types:
                trial_type_fraction_array = np.empty((len(self.counts),self.number_of_types))
                trial_type_count_array = np.empty((len(self.counts),self.number_of_types))
                for i in range(len(self.counts)):
                    trial_type_fraction_array[i,:], trial_type_count_array[i,:] = self.randomize_droplet_type(
                        self.type_fractions[i], self.counts[i])
                    #APA

                self.type_count_MC_matrix_at_measured[trial] = trial_type_count_array
                self.type_fraction_MC_matrix_at_measured[trial] = trial_type_fraction_array

            trial_T_array = [self.calculate_T(
                trial_t_array[i], trial_d) for i in range(len_calculated)]

            self.t_MC_matrix_at_measured[trial, :] = trial_t_array
            self.T_MC_matrix_at_measured[trial, :] = trial_T_array
            self.v_MC_matrix_at_measured[trial, :] = trial_v_array
            self.volume_MC_array_at_measured[trial] = trial_volume
            self.fractions_MC_matrix_at_measured[trial, :] = trial_f_array
            self.distances_matrix_at_measured[...] = self.measured_distances

    def run_MC_for_J(self, trials=1000):
        '''        
        In each trial, A value is sampled for q,d and s

        t and T errors can then be calculated for all points

        Then arrays for t,T and J are calculated

        '''

        self.number_of_trials = trials

        len_calculated = len(self.calculated_distances)

        self.J_MC_matrix = np.zeros([trials, len_calculated])
        self.t_MC_matrix = np.zeros([trials, len_calculated])
        self.T_MC_matrix = np.zeros([trials, len_calculated])
        self.v_MC_matrix = np.zeros([trials, len_calculated])
        self.distances_matrix = np.zeros([trials, len_calculated])

        for trial in range(trials):

            trial_q = self.q_value
            trial_d = np.random.normal(loc=self.d_value, scale=self.d_sigma)
            trial_s = np.random.normal(loc=self.s_value, scale=self.s_sigma)

            trial_f_array = [np.divide(np.random.binomial(
                self.counts[i], self.fractions[i]), self.counts[i]) for i in range(len(self.counts))]

            trial_t_array = [self.calculate_t(self.calculated_distances[i],
                                              trial_d,
                                              trial_q,
                                              trial_s) for i in range(len_calculated)]

            trial_v_array = [self.calculate_v(
                trial_d, trial_q, trial_s) for i in range(len_calculated)]

            trial_T_array = [self.calculate_T(
                trial_t_array[i], trial_d) for i in range(len_calculated)]

            trial_J_array = [self.calculate_J(trial_q,
                                              trial_s,
                                              trial_d,
                                              self.measured_distances[i+1],
                                              self.measured_distances[i],
                                              trial_f_array[i+1],
                                              trial_f_array[i]) for i in range(len_calculated)]

            self.J_MC_matrix[trial, :] = trial_J_array
            self.t_MC_matrix[trial, :] = trial_t_array
            self.T_MC_matrix[trial, :] = trial_T_array
            self.v_MC_matrix[trial, :] = trial_v_array

            self.distances_matrix[...] = self.calculated_distances

    def generate_plotting_label(self):
        self.plot_label = '{0}\n{1:.2f} um\n{2} kHz\n{3:.2f} ul/min\n{4} m/s'.format(self.date, self.d_value*1e6,
                                                                                     self.frequency, self.q_value, self.ideal_v)

    def save_column_file(self):

        description_line = ' '.join(['Distance',
                                     'Time',
                                     'Time_err_low',
                                     'Time_err_high',
                                     'Temperature',
                                     'Temperature_err_low',
                                     'Temperature_err_high',
                                     'J',
                                     'J_err_low',
                                     'J_err_high'])

        data = np.column_stack((self.calculated_distances,
                                np.array(self.ideal_t_at_calculated),
                                self.low_t_err,
                                self.high_t_err,
                                np.array(self.ideal_T_at_calculated),
                                self.low_T_err,
                                self.high_T_err,
                                np.array(self.ideal_J),
                                self.low_J_err,
                                self.high_J_err))

        np.savetxt(self.date + '_interpolatedpos.txt', data,
                   header=description_line, fmt='%.4e')

        description_line = ' '.join(['Distance',
                                     'Time',
                                     'Time_err_low',
                                     'Time_err_high',
                                     'Temperature',
                                     'Temperature_err_low',
                                     'Temperature_err_high'])

        data = np.column_stack((self.measured_distances,
                                np.array(self.ideal_t_at_measured),
                                self.low_t_err_at_measured,
                                self.high_t_err_at_measured,
                                np.array(self.ideal_T_at_measured),
                                self.low_T_err_at_measured,
                                self.high_T_err_at_measured))

        np.savetxt(self.date + '_measuredpos.txt', data,
                   header=description_line, fmt='%.4e')

    def plot_measured_T_vs_t(self, ax=None, **kwargs):
        if ax == None:
            plt.figure()
            ax = plt.gca()
        xerr = np.array([self.low_t_err_at_measured,
                         self.high_t_err_at_measured])
        yerr = np.array([self.low_T_err_at_measured,
                         self.high_T_err_at_measured])
        ax.scatter(self.ideal_t_at_measured,
                   self.ideal_T_at_measured,
                   marker='x',
                   **kwargs)
        kwargs['label'] = None
        ax.errorbar(self.ideal_t_at_measured,
                    self.ideal_T_at_measured,
                    xerr=xerr,
                    yerr=yerr,
                    fmt='none',
                    capsize=5,
                    **kwargs)

    def plot_T_vs_measured_distance(self, ax=None, **kwargs):
        if ax == None:
            plt.figure()
            ax = plt.gca()
        yerr = np.array([self.low_T_err_at_measured,
                         self.high_T_err_at_measured])
        ax.scatter(self.measured_distances,
                   self.ideal_T_at_measured,
                   marker='x',
                   **kwargs)
        kwargs['label'] = None
        ax.errorbar(self.measured_distances,
                    self.ideal_T_at_measured,
                    yerr=yerr,
                    fmt='none',
                    capsize=5,
                    **kwargs)

    def plot_calculated_T_vs_t(self, ax=None, **kwargs):
        if ax == None:
            plt.figure()
            ax = plt.gca()
        xerr = np.array([self.low_t_err, self.high_t_err])
        yerr = np.array([self.low_T_err, self.high_T_err])

        ax.scatter(self.ideal_t_at_calculated,
                   self.ideal_T_at_calculated,
                   marker='x',
                   **kwargs)
        kwargs['label'] = None
        ax.errorbar(self.ideal_t_at_calculated,
                    self.ideal_T_at_calculated,
                    xerr=xerr,
                    yerr=yerr,
                    fmt='none',
                    capsize=5,
                    **kwargs)

    def plot_J_vs_T(self, ax=None, **kwargs):
        if ax == None:
            plt.figure()
            ax = plt.gca()
        xerr = np.array([self.low_T_err, self.high_T_err])
        yerr = np.array([self.low_J_err, self.high_J_err])

        ax.scatter(self.ideal_T_at_calculated,
                   self.ideal_J,
                   **kwargs)
        kwargs['label'] = None
        ax.errorbar(self.ideal_T_at_calculated,
                    self.ideal_J,
                    xerr=xerr,
                    yerr=yerr,
                    fmt='none',
                    capsize=2, lw=1,
                    **kwargs)

    def plot_J_vs_distance(self, ax=None, **kwargs):
        if ax == None:
            plt.figure()
            ax = plt.gca()
        yerr = np.array([self.low_J_err, self.high_J_err])

        ax.scatter(self.calculated_distances,
                   self.ideal_J,
                   **kwargs)
        kwargs['label'] = None
        ax.errorbar(self.calculated_distances,
                    self.ideal_J,
                    yerr=yerr,
                    fmt='none',
                    capsize=5,
                    **kwargs)