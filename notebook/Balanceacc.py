@keras_export('keras.metrics.Balanceacc')
class Balanceacc(Metric):
  def __init__(self,
               thresholds=None,
               top_k=None,
               class_id=None,
               name=None,
               dtype=None):
    super(Balanceacc, self).__init__(name=name, dtype=dtype)
    self.init_thresholds = thresholds
    self.top_k = top_k
    self.class_id = class_id

    default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
    self.thresholds = metrics_utils.parse_init_thresholds(
        thresholds, default_threshold=default_threshold)

    self.true_positives = self.add_weight(
        'true_positives',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)

    self.true_negatives = self.add_weight(
        'true_negatives',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)
    
    self.false_positives = self.add_weight(
        'false_positives',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)
    
    self.false_negatives = self.add_weight(
        'false_negatives',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)

  def update_state(self, y_true, y_pred, sample_weight=None):
    return metrics_utils.update_confusion_matrix_variables(
        {
            metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
            metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,
            metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
            metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives
        },
        y_true,
        y_pred,
        thresholds=self.thresholds,
        top_k=self.top_k,
        class_id=self.class_id,
        sample_weight=sample_weight)

  def result(self):
    sen = math_ops.div_no_nan(
        self.true_positives,
        math_ops.add(self.true_negatives, self.false_positives))

    spe = math_ops.div_no_nan(
        self.true_negatives,
        math_ops.add(self.true_negatives, self.false_positives))

    result = math_ops.multiply(tf.constant([0.5]), math_ops.add(sen, spe))
    
    
    return result[0] if len(self.thresholds) == 1 else result

  def reset_states(self):
    num_thresholds = len(to_list(self.thresholds))
    K.batch_set_value(
        [(v, np.zeros((num_thresholds,))) for v in self.variables])

  def get_config(self):
    config = {
        'thresholds': self.init_thresholds,
        'top_k': self.top_k,
        'class_id': self.class_id
    }
    base_config = super(Balanceacc, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))