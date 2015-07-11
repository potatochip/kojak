# Fitting a feature selector
def feature_selection(train_instances):
    log_info('Crossvalidation started... ')
    selector = VarianceThreshold()
    selector.fit(train_instances)
    log_info('Number of features used... ' +
              str(Counter(selector.get_support())[True]))
    log_info('Number of features ignored... ' +
              str(Counter(selector.get_support())[False]))
    return selector
 
#Learn the features to filter from train set
fs = feature_selection(train_instances)
 
#Transform train and test subsets
train_instances = fs.transform(train_instances)
test_instances = fs.transform(test_instances)