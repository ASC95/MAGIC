#!/usr/bin/env python

AMPDS_START_DATE = '2012-04-01 07:00:00-00:00'
AMPDS_END_DATE = '2014-04-01 06:59:00-00:00'

# Mapping of appliance names to AMPds2 meter IDs
METER_MAPPING = {
    'house': 'meter1', 'light2': 'meter2', 'light3': 'meter3', 'light4': 'meter4',
    'dryer': 'meter5', 'washer': 'meter6', 'sockets7': 'meter7', 'dishwasher': 'meter8',
    'workbench': 'meter9', 'security': 'meter10', 'fridge': 'meter11', 'hvac': 'meter12',
    'garage': 'meter13', 'heat_pump': 'meter14', 'water_heater': 'meter15',
    'light16': 'meter16', 'sockets17': 'meter17', 'rental_suite': 'meter18',
    'television': 'meter19', 'sockets20': 'meter20', 'oven': 'meter21'
}