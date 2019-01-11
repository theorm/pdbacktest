import array, io
import backtrader as bt
import backtrader.plot
import numpy as np

def get_signal_from_sentiment(sentiment, boundaries=[-1, 1]):
  low_boundary, high_boundary = boundaries
  
  for v in sentiment:
    if v >= high_boundary:
      yield -1 # Go short
    elif v <= low_boundary:
      yield 1 # Go long
    else:
      yield 0

def plot_cerebro(cerebro, filename=None, width_inch=16, height_inch=9):
  '''
  From `Cerebro.plot`
  '''
  numfigs = 1
  iplot = False
  start = None
  end = None
  use = None

  plotter = bt.plot.Plot(numfigs=numfigs, iplot=iplot, start=start, end=end,
                         width=width_inch, height=height_inch, dpi=300, tight=False, use=use, style='line') #, style='candle'

  figs = []
  for stratlist in cerebro.runstrats:
    for si, strat in enumerate(stratlist):
      rfig = plotter.plot(strat, figid=si * 100,
                          numfigs=numfigs, iplot=iplot,
                          start=start, end=end, use=use)
      figs.append(rfig)
  
  if filename is not None:
    plotter.savefig(figs[0][0], filename)
    plotter.mpyplot.close(figs[0][0])
    return filename
  else:
    plotter.show()
    return plotter

def backtest(ohlc, sentiment, signal_boundaries=[-1, 1], instruments = [], cash=None, name=None, filename=None, plot_to_buffer=False, width_inch=16, height_inch=9):
  '''
  Arguments:
    * ohlc - a pandas DataFrame with OHLC ('Open', 'High', 'Low', 'Close')
    * sentiment - a list or numpy array with sentiment values. 
    * signal_boundaries - a tuple of lower and upper boundary that trigger "long" and "short" signals 
  '''
  ohlc_len = len(ohlc)
  sentiment_len = len(sentiment)
  assert ohlc_len == sentiment_len, 'Expected OHLC length {} to be the same as sentiment length {}'.format(ohlc_len, sentiment_len)
  
  instruments_indicators = []
  
  for i in instruments:
    instrument_name = i[0]
    sanitised_instrument_name = instrument_name.lower().replace(' ', '_')
    instrument_values = i[1]
    instrument_values_len = len(instrument_values)
    assert ohlc_len == sentiment_len, \
      'Expected OHLC length {} to be the same as instrument {} length {}'.format(ohlc_len, instrument_name, instrument_values_len)  
    
    is_oscillator = i[2] if len(i) == 3 else False
  
    class _Indicator(bt.Indicator):
      lines = (sanitised_instrument_name,)
      plotinfo=dict(
        plotname=instrument_name,
        subplot=is_oscillator
      )
      def __init__(self):
        getattr(self.lines, sanitised_instrument_name).array = array.array('d', instrument_values)
    
    _Indicator.__name__ = '_Indicator_{}'.format(sanitised_instrument_name)
    instruments_indicators.append(_Indicator)
    
    
  signal = list(get_signal_from_sentiment(sentiment, signal_boundaries))
  long_signal_indices = np.array(signal)[np.where(np.array(signal) == 1)]
  if len(long_signal_indices) > 0:
    first_long_index = long_signal_indices[0]
    first_price = ohlc['Close'].values[first_long_index]
  else:
    first_price = None

  class _SignalAndSentimentIndicator(bt.Indicator):
    lines = ('signal', 'sentiment')
    plotinfo=dict(
      plothlines=signal_boundaries
    )
    plotlines=dict(
      signal=dict(_plotskip=True),
    )
  
    def __init__(self):
      self.lines.sentiment.array = array.array('d', sentiment)
      self.lines.signal.array = array.array('d', signal)
      
  class _BtSignalStrategy(bt.SignalStrategy):
    def __init__(self):
      for indicator in instruments_indicators:
        indicator()

      sas = _SignalAndSentimentIndicator()
      self.signal_add(bt.SIGNAL_LONG, sas.signal)
      self.signal_add(bt.SIGNAL_LONGEXIT, sas.signal)
        
  data = bt.feeds.PandasData(dataname=ohlc)
  
  if name is not None:
    data._name = name
  else:
    data._name = getattr(ohlc, '_symbol') if hasattr(ohlc, '_symbol') else None
  
  cerebro = bt.Cerebro()
  cerebro.addstrategy(_BtSignalStrategy)
  cerebro.adddata(data)
  
  if cash is not None:
    cerebro.broker.setcash(cash)
  elif first_price is not None:
    cerebro.broker.setcash(first_price)

  # XXX: If set to 100%, the broker will reject orders where open>close from previous bar
  # cerebro.addsizer(bt.sizers.PercentSizer, percents=50)
  cerebro.addsizer(bt.sizers.SizerFix, stake=1)

  cerebro.addanalyzer(bt.analyzers.SharpeRatio)
  cerebro.addanalyzer(bt.analyzers.Returns)
  cerebro.addanalyzer(bt.analyzers.SQN)
  cerebro.addanalyzer(bt.analyzers.DrawDown)

  thestrats = cerebro.run()
  thestrat = thestrats[0]
  
  stats = {}
  for n in thestrat.analyzers.getnames():
    stats[n] = getattr(thestrat.analyzers, n).get_analysis()

  if plot_to_buffer:
    buf = io.BytesIO()
    plot_result = plot_cerebro(cerebro, filename=buf, width_inch=width_inch, height_inch=height_inch)
    buf.seek(0)
  else:
    plot_result = plot_cerebro(cerebro, filename=filename, width_inch=width_inch, height_inch=height_inch)
  
  return stats, plot_result