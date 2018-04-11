# fix the price json files, output valid json with BTC information only
import sys

if len(sys.argv) != 3:
	print('usage: python get-btc-json.py <json-in> <json-out>')

starting = True
with open(sys.argv[1]) as f_in, open(sys.argv[2], 'a') as f_out:
	f_out.write('[')
	for line in f_in:
		if starting:
			starting = False
		else:
			f_out.write(',\n')
		time_end_pos = line.find(':::')
		f_out.write('{"time":\"'+line[:time_end_pos]+'\",')
		BTC_start_pos = line.find('{')
		BTC_end_pos = line.find('}')
		f_out.write(line[BTC_start_pos+1:BTC_end_pos+1])
	f_out.write(']')

