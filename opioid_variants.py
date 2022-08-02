import requests, sys
import json
from pathlib import Path
import numpy as np

variants = open('data_variants/oud_od_variants.txt').read().split('\n')

populations = dict(
  AFR = ['ACB', 'ASW', 'ESN', 'GWD', 'LWK', 'MSL', 'YRI'],
  EUR = ['CEU', 'FIN', 'GBR', 'IBS', 'TSI']
)

# Create population directories
def create_dirs(populations):
  for pop in populations:
    for subpop in populations[pop]:
      Path("data_variants/{pop}/{subpop}".format(
        pop=pop, subpop=subpop
      )).mkdir(parents=True, exist_ok=True)
 
server = "https://rest.ensembl.org"
ext_find_pop = "/info/variation/populations/homo_sapiens?filter=LD"

# Get proxy variants
def get_variants(variants, populations, d, win_size, r2):
  for pop in populations:
    print('Population:', pop)
    for subpop in populations[pop]:
      print('Subpopulation:', subpop)
      for i in range(len(variants)):
        ext_find_proxy = "/ld/human/{0}/1000GENOMES:phase_3:{1}?d_prime={2};window_size={3};r2={4}".format(
                          variants[i], subpop, d, win_size, r2)
        r = requests.get(server+ext_find_proxy, headers={ "Content-Type" : "application/json"})
        if not r.ok:
          r.raise_for_status()
          sys.exit()
        decoded = r.json()
        f = open("data_variants/{pop}/{subpop}/variants_in_ld_with_{var}.json".format(
          pop=pop, subpop=subpop, var=variants[i]), "w")
        f.write(json.dumps(decoded))
        f.close()

def get_proxy(pop, subpop, var):
  with open("data_variants/{pop}/{subpop}/variants_in_ld_with_{var}.json".format(
          pop=pop, subpop=subpop, var=var), "r") as read_file:
    data = json.load(read_file)
  proxy_variants = [data[j]['variation2'] for j in range(len(data))]
  return np.asarray(proxy_variants)

def retrieve_proxies(populations, variants):
  all_proxies = np.array([])
  for pop in populations:
    print('Population:', pop)
    for subpop in populations[pop]:
      print('Subpopulation:', subpop)
      for i in range(len(variants)):
        data = get_proxy(pop, subpop, variants[i])
        all_proxies = np.concatenate((all_proxies, data))
  return np.unique(all_proxies)

proxy_variants = retrieve_proxies(populations, variants)
# get_variants(variants, populations, 1.0, 300, 0.8)

# Write text file with the retrieved proxy variants
def output_proxies(filename, data):
  textfile = open(filename, "w")
  for element in data:
      textfile.write(element + "\n")
  textfile.close()

output_proxies('proxy_variants.txt', proxy_variants)