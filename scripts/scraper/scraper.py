from splinter import Browser
from time import sleep
from os import listdir
from re import sub
import sys
import os
import tempfile

from multiprocessing import Pool

def scraper():

    #section-hero-header-title-subtitle

    if chrome.is_element_present_by_css('div[class="section-result-header-container"]'):

        div_count = len(chrome.find_by_xpath("//div[contains(@jsaction, 'pane.resultSection.click;"
                                             "keydown:pane.resultSection.keydown;mouseover:pane.resultSection.in;"
                                             "mouseout:pane.resultSection.out;focus:pane.resultSection.focusin;"
                                             "blur:pane.resultSection.focusout')]"))
        for count in range(0, div_count):

        #chrome.find_by_css('div[class="section-result-header-container"]:first-of-type').click()

        # if chrome.is_element_not_present_by_xpath('//*[@id="pane"]/div/div[1]/div/div/div[2]/div[1]/div[1]/h1'):
        #     chrome.find_by_xpath("//div[contains(@jsaction, 'pane.resultSection.click;"
        #                          "keydown:pane.resultSection.keydown;mouseover:pane.resultSection.in;"
        #                          "mouseout:pane.resultSection.out;focus:pane.resultSection.focusin;"
        #                          "blur:pane.resultSection.focusout')]")[1].click()

        # print(chrome.find_by_xpath('//*[@id="pane"]/div/div[1]/div/div/div[2]/div[1]/div[1]/h1').text)
        # print(chrome.find_by_css('div[class="section-hero-header-title-subtitle"] ::text').first.value)
        # print(chrome.find_by_xpath('//*[@id="pane"]/div/div[1]/div/div/div[2]').text)

            try:

                chrome.find_by_xpath("//div[contains(@jsaction, 'pane.resultSection.click;"
                                     "keydown:pane.resultSection.keydown;mouseover:pane.resultSection.in;"
                                     "mouseout:pane.resultSection.out;focus:pane.resultSection.focusin;"
                                     "blur:pane.resultSection.focusout')]")[count].click()
                sleep(2)

                try:
                    name = chrome.find_by_xpath('//*[@id="pane"]/div/div[1]/div/div/div[2]/div[1]/div[1]/h1').text
                except:
                    name = place
                try:
                    # get category
                    category = chrome.find_by_xpath(
                        "//*[@id='pane']/div/div[1]/div/div/div[2]/div[1]/div[2]/div/div[2]/span[1]/span[1]/button").text
                except:
                    try:
                        category = chrome.find_by_xpath('//*[@id="pane"]/div/div[1]/div/div/div[2]/div[1]/h2/span').text
                    except:
                        category = chrome.find_by_xpath(
                            '/html/body/jsl/div[3]/div[9]/div[8]/div/div[1]/div/div/div[3]/div[1]/h2/span').text

                geocode_url = chrome.url.split("@")
                try:
                    geocode_arr = geocode_url[1].split("!3d")
                    geocode_arr = geocode_arr[1].split("!4d")
                    # geocode_arr = geocode_url[1].split(",")
                    lat = geocode_arr[0]
                    lng = geocode_arr[1]
                except:
                    try:
                        geocode_arr = geocode_url[2].split("!3d")
                        geocode_arr = geocode_arr[1].split("!4d")
                        # geocode_arr = geocode_url[1].split(",")
                        lat = geocode_arr[0]
                        lng = geocode_arr[1]

                    except:
                        try:
                            geocode_arr = geocode_url[1].split(",")
                            lat = geocode_arr[0]
                            lng = geocode_arr[1]
                        except:
                            geocode_arr = geocode_url[2].split(",")
                            lat = geocode_arr[0]
                            lng = geocode_arr[1]

                print(name.strip() + "\t" + category.strip() + "\t" + str(lat) + "\t" + str(lng))

                chrome.visit("https://www.google.com/maps/search/" + place_in_link.strip() + "/@{},{}".format(places_arr[place][1][0],places_arr[place][1][1]))

                sleep(5)

            except:
                continue

    elif chrome.is_element_present_by_xpath('//*[@id="pane"]/div/div[1]/div/div/div[2]/div[1]/div[1]/h1'):
        # print(chrome.find_by_xpath('//*[@id="pane"]/div/div[1]/div/div/div[2]/div[1]/div[1]/h1').text)
        # print(chrome.find_by_xpath('//*[@id="pane"]/div/div[1]/div/div/div[2]/div[1]/h2/span').text)
        # print(chrome.find_by_xpath('//*[@id="pane"]/div/div[1]/div/div/div[2]').text)

        try:
            name = chrome.find_by_xpath('//*[@id="pane"]/div/div[1]/div/div/div[2]/div[1]/div[1]/h1').text
        except:
            name = place
        try:
            # get category
            category = chrome.find_by_xpath(
                "//*[@id='pane']/div/div[1]/div/div/div[2]/div[1]/div[2]/div/div[2]/span[1]/span[1]/button").text
        except:
            try:
                category = chrome.find_by_xpath('//*[@id="pane"]/div/div[1]/div/div/div[2]/div[1]/h2/span').text
            except:
                category = chrome.find_by_xpath(
                    '/html/body/jsl/div[3]/div[9]/div[8]/div/div[1]/div/div/div[3]/div[1]/h2/span').text

        geocode_url = chrome.url.split("@")
        try:
            geocode_arr = geocode_url[1].split("!3d")
            geocode_arr = geocode_arr[1].split("!4d")
            # geocode_arr = geocode_url[1].split(",")
            lat = geocode_arr[0]
            lng = geocode_arr[1]
        except:
            try:
                geocode_arr = geocode_url[2].split("!3d")
                geocode_arr = geocode_arr[1].split("!4d")
                # geocode_arr = geocode_url[1].split(",")
                lat = geocode_arr[0]
                lng = geocode_arr[1]

            except:
                try:
                    geocode_arr = geocode_url[1].split(",")
                    lat = geocode_arr[0]
                    lng = geocode_arr[1]
                except:
                    geocode_arr = geocode_url[2].split(",")
                    lat = geocode_arr[0]
                    lng = geocode_arr[1]

        print(name.strip() + "\t" + category.strip() + "\t" + str(lat) + "\t" + str(lng))


def parse(f):
    working_f = open(f,'r')
    bbox = []
    name = []
    type = []
    locations = []
    loc = []
    # vp = 0
    geo = 0
    for line in working_f:
        # if 'viewport' in line:
        #     vp = 1
        # elif vp >= 1:
        #     if vp == 2 or vp == 3 or vp == 6 or vp == 7:
        #         val = float(line.split(':')[-1].strip().strip(','))
        #         viewPort += [val]
        #     vp += 1
        #     if vp == 8:
        #         vp = 0
        #         bbox += [viewPort]
        #         viewPort = []

        # Gets the building location point
        if 'geometry' in line:
            geo = 1
        elif geo >= 1:
            if geo == 2 or geo == 3:
                val = float(line.split(':')[-1].strip().strip(','))
                loc += [val]
            geo += 1
            if geo == 4:
                locations += [loc]
                geo = 0
                loc = []

        # Gets the building name
        if '"name"' in line:
            new_name = line.split(':')[-1].split(',')[0].replace('"', "").strip()
            name += [new_name]

        # gets the types
        elif 'types' in line:
            new_types = line.split(':')[-1].strip().replace('[', '').replace(']', '').replace('"', "").strip().split(
                ',')[:-1]
            type += [new_types]

    return name,type,locations#bbox,name,type,locations



if __name__ == '__main__':
    api_hit = '/Users/devinjmcconnell/Documents/Research/Opioid-LifeRhythm/data/api_hit/'
    dirs = listdir(api_hit)
    dirs = [d for d in dirs if os.path.isdir(api_hit + d)]
    fs = [api_hit + item + '/' + f for item in dirs for f in listdir(api_hit + item)]

    name = []
    type = []
    locations = []
    for item in fs:
        # b, n, t, l = parse(item)
        n, t, l = parse(item)
        name += [n]
        type += [t]
        locations += [l]

    data = {}

    for i in range(len(name)):
        for j in range(len(name[i])):
            if name[i][j] not in data:
                data[name[i][j]] = [type[i][j], locations[i][j]]

    places_arr = data
    chrome = Browser("chrome", executable_path='/usr/local/bin/chromedriver')
    sleep(20)
    print("Name\tCategory\tLatitude\tLongitude\n")
    for place in places_arr:
        place = place.strip()

        # print place
        place_in_link = sub(r"\s+", '+', place)

        chrome.visit(
            "https://www.google.com/maps/search/" + place_in_link.strip() + "/@{},{}".format(places_arr[place][1][0],
                                                                                             places_arr[place][1][1]))
        # chrome.visit("https://www.google.com/maps/search/Service+Plus+Plumbing/@41.6346268,-72.2192457")
        # chrome.visit("https://www.google.com/maps/search/Radiology/@40.6624355,-73.6479281")
        # chrome.visit('https://www.google.com/maps/search/Favela+Encantada+@+Candibar/@42.3500654,-71.0654299')
        sleep(2)

        print("https://www.google.com/maps/search/" + place_in_link.strip() + "/@{},{}".format(places_arr[place][1][0],
                                                                                               places_arr[place][1][1]),
              file=sys.stderr)
        scraper()
    chrome.quit()

#'https://www.google.com/maps/place/Dunkin'/@41.8258278,-72.1608234,12z/data=!4m8!1m2!2m1!1sdunkin+donuts!3m4!1s0x89e68f2fbe3726cd:0xe309a66cff34c04f!8m2!3d41.8567697!4d-72.1795978'