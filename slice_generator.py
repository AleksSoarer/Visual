import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import os
import math
from cycler import cycler
from itertools import chain
import json

#def load_global_data():
#    global license_name
#    global river
#    global slice_name
#    global material_df
#    global types_df
#    global slice_data
#    global file_adress
#    global error

    #читаем таблицу материалов из файла
material_df = pd.read_csv('materials_db.csv', sep=';')

    #создаём датафрейм для обработки материалов по типам
types_df = pd.DataFrame(columns=['prs', 'torfa', 'rud_plast', 'plotik'])

    #создаём базовый датафрейм для данных о скважинах исключаем 'name', 'river', 
slice_data = pd.DataFrame(columns = ['number', 'x', 'y', 'z', 'layers_power', 'num_of_layers','layers_id'])

    #создаём глобальные переменные для формирования адреса
license_name = ''
river = ''
slice_name = ''
error = 0
file_adress = ''

#########################################__ОПРЕДЕЛЯЕМ_ФУНКЦИИ__############################################################

#подсчёт слоёв одного типа
def one_type_layer_calc3(type_elem):    #ver3 для подсчёта по длинне датафрйма а не количеству файлов

    
    out_layers_list = []
    tmp_layers_list = []
    type_all = []

    for i in range(0 ,len(slice_data)):

        if types_df[type_elem][i] not in type_all:
                    
            if types_df[type_elem][i] == 6:
                types_df[type_elem][i] = []
            type_all.append(types_df[type_elem][i])
    
    max_item = []

    for item in type_all:
        #print(len(item))
        if len(max_item ) < len(item):
            max_item = item

    

    flat_list_1 = max_item
    #(list(chain.from_iterable(max_item)))
    flat_list_2 = (list(chain.from_iterable(type_all)))

    
    #print(flat_list_1, flat_list_2, '111')
    for elem in flat_list_2:
        if elem not in flat_list_1:
            flat_list_1.append(elem)

    #print(flat_list_1)
    return flat_list_1


#Функция для генерации списка присутствующих материалов
#Создадим датафрейм на основе layers_id 
#разделяем материалы скважин по типам для дальнейшей обработки
def types_agregator(i):
    
    
    full = []     #итоговый список списков по типам и меатериалам в дф!
    material_category = []   #все материалы скважины
    mc2 = []    #временный список, куда складываем материалы одного типа, пока не призойдёт смена типа. Тогда его содержимое передаем дальше а список обнуляем

    
    slice_data_layer = slice_data['layers_id'][i]

    for item, types in enumerate(slice_data_layer):

    
        if item == 0: 
            material_category.append(types)
            mc2.append(types)
            types_df.at[i, material_df.iloc[types][2]] = mc2
        
        
        elif 0 < item : #< (len(slice_data_layer)-0) 

            #print('номер=', item, 'id', types, 'тип', material_df.iloc[types][2], '!!', material_df.iloc[material_category[item-1]][2])


            if (material_df.iloc[types][2] != material_df.iloc[material_category[item-1]][2]) and ((item + 1) < len(slice_data_layer)):
                full.append(mc2)
                mc2 = []
                material_category.append(types)
                mc2.append(types)

            elif (item + 1) == len(slice_data_layer):
                full.append(mc2)
                mc2 = []
                material_category.append(types)
                mc2.append(types)
                full.append(mc2)

                #print('mc2', mc2)


            else:
                material_category.append(types)
                mc2.append(types)

            #print('текущий материал',mc2, material_df.iloc[types][2])
            types_df.at[i, material_df.iloc[types][2]] = mc2

    

#генератор слоёв для скважин
def layer_generator(power, layer_id, project_layers=None):
    
    
    c = dict(zip(layer_id, power))
    out_layer = []
    
    for item in project_layers:
        temp = c.get(item, 0)
        out_layer.append(temp)
        
    return out_layer


#здесь считаем расстояние на плоскости между скважинами
def step_calculator(x, y):
    

    x0 = 0
    y0 = 0
    shift_list=[]
    row_list=[]
    
    
    for step in range(0, slice_data.shape[0]):
        shift = round(((x[step]-x0)**2 + (y[step]-y0)**2)**0.5, 1)
        shift_list.append(shift)
        row_list.append(round(shift))
        x0, y0 = x[step], y[step]
        
        

    shift_list[0]=0
    row_list[0]=0
    
    
    step = []
    #table_step = []
    stack = 0
    for item in shift_list:
        #table_step.append(round(item,2))
        
        stack+=item
        step.append(stack)


    return step, row_list


#генерируем таблицу толщин слоёв
def layers_power_generator():
    
    
    layers = pd.DataFrame()
    for i in range(0, len(project_layers)):                         #по количеству слоёв
        sld = []
        for j in range(0, slice_data.shape[0]):
            sld.append(slice_data['full_layers'][j][i])
        layers[i] = sld
    

    return layers



#Детектор перехода слоёв
def transition_detector(id_1, types): #=None
    
    #алгоритм сопряжения материалов в слоях в зависимости от типа материала
    bh_ind = slice_data['number'][id_1]
    #print('входящий номер по списку', id_1, '\nФактический номер скважины', bh_ind)
    
    ####types = ['prs', 'torfa', 'torfa', 'torfa', 'torfa', 'torfa', 'rud_plast', 'rud_plast', 'plotik', 'plotik']
    ####types = передаются снаружи. Спасок создаёт одельная функция. 
    #Смотрим по полному набору слоёв, исходя из него добавляем в список его тип. Типы могут идти вразнобой, все зависит от фактических данных со скважины
    
    all_layers = [ x for x in range(len(types))] #все номера слоёв
    
    a_1 = slice_data['full_layers'][id_1]  #первая реальная скважина для генерации сопряжения
    #print('Первая скважина', a_1)
    
    a_2 = slice_data['full_layers'][id_1+1]  #вторая реальная скважина для генерации сопряжения
    #print('Вторая скважина', a_2)
    
    transition_list = []   #список переходов для сопряжений материалов
    
    for elem in range(0,len(types)-1):        #проходим по всем слоям и сравниваем со следующим
                                            #кроме последнего, ему объединяться не с чем
        
    
        #проверяем по условию происходит ли смена материала, то есть переход толщины слоя через 0
        
        if (a_1[elem] != a_2[elem]) and (a_1[elem] == 0 or a_2[elem] == 0):  
            #print(types[elem], 'Происходит смена толщины материала', types[elem], a_1[elem],'на',a_2[elem])
            type_for_calculate = types[elem]
            
            
            index_mat_layers = [i for i, v in enumerate(types) if v == type_for_calculate]
            #print('список исследуемого слоя', index_mat_layers)
            transition_list.append(index_mat_layers)
                
    transition = []
    [transition.append(x) for x in transition_list if x not in transition] #список переходов
    
    tansition_layer_list = [x for group in transition for x in group] #список слоёв учавствующих в переходах
    
    no_transition = []
    
    [no_transition.append(x) for x in all_layers if x not in tansition_layer_list] #слои не участвующие в переходе
    
    
    if len(tansition_layer_list) > 0: #если есть слои с переходом - вызываем генератор виртуальных скважин
        
        virtual_bh_generator(id_1, a_1, a_2, bh_ind, all_layers, transition, no_transition, types)
    
    


#генератор виртуальных скважин для определения мест перехода материала
def virtual_bh_generator(id_1, a_1, a_2, bh_ind, all_layers, transition, no_transition, types=None):
    
    
    #types = ['prs', 'torfa', 'torfa', 'torfa', 'torfa', 'torfa', 'rud_plast', 'rud_plast', 'plotik', 'plotik']
    
    a_1 = slice_data['full_layers'][id_1]
    a_2 = slice_data['full_layers'][id_1+1]
    
    #готовим загтовки под виртуальные скважины
    a_13 = [0 for x in range(len(types))]
    a_15 = [0 for x in range(len(types))]
    a_16 = [0 for x in range(len(types))]
    
    
    transition_koeff = []
    
    for transition_list in transition:
        
        
        first_num = 0                 #толщина первой скважины по конкретному материалу
        second_num = 0                #толщина второй скважины по конкретному материалу
        
        for item in transition_list:
            
            
            first_num = first_num + a_1[item] 
            second_num = second_num + a_2[item]
            
        
            
        koeff = (first_num + second_num)/2
        transition_koeff.append(koeff)
        
        if first_num == 0:
            koeff_1 = 0
        else:
            koeff_1 = koeff / first_num
        
        
        if second_num == 0:
            koeff_2 = 0
        else:
            koeff_2 = koeff / second_num
        
        for item in transition_list:
            a_13[item] = a_1[item]*koeff_1
            a_16[item] = a_2[item]*koeff_2
            
        
    
    for item in no_transition:         #добавим среднее значение виртуальных скважин, где нет перехода
        a_13[item] = a_15[item] = a_16[item] = (a_1[item]+a_2[item])/2 
        #print('\nНомер слоя без перехода', item, 'контроль номера и элемента слоя',a_16[item], '\n')
    
    
    #общие координаты вируальных скважин
    xx = (slice_data['x'][id_1] + slice_data['x'][id_1+1])/2
    yy = (slice_data['y'][id_1] + slice_data['y'][id_1+1])/2
    zz = (slice_data['z'][id_1] + slice_data['z'][id_1+1])/2
    
    #name = slice_data['name'][0]
    #river = slice_data['river'][0]
    #license_name = slice_data['license'][0]
    
    
    #add left row to end of DataFrame  name, river,
    #print('добавляем первую в список на позицию',len(slice_data.index))
    slice_data.loc[ len(slice_data.index )] = [
        (bh_ind+0.1), xx, yy, zz, a_13, 
        0, slice_data['layers_id'][id_1], 
        a_13
    ]


    #add center row to end of DataFrame  name, river,
    #print('добавляем вторую в список на позицию',len(slice_data.index+1))
    slice_data.loc[ len(slice_data.index + 1 )] = [
        (bh_ind+0.2), xx, yy, zz, a_15, 
        0, slice_data['layers_id'][id_1], #15 = len(layers_power)
        a_15 
    ]


    #add rigth row to end of DataFrame name, river,
    #print('добавляем центр в список на позицию',len(slice_data.index+2),'\n\n\n#########################\n\n')
    slice_data.loc[ len(slice_data.index + 2 )] = [
        (bh_ind+0.3), xx, yy, zz, a_16, 
        0, slice_data['layers_id'][id_1], 
        a_16 
    ]
            

#загрузчик из списка словарей скважин разреза в датафрейм 
def data_extractor(list_of_bh):

    global slice_data
    global license_name
    global river
    global slice_name
    
    
    for item in list_of_bh:
        
        borehole_number = number_converter(item['borehole_number'])
        gps = item['gps']
        layers_power = item['layers_power']
        pre_layers_id = item['layers_id']
        layers_id = []
        for _ in pre_layers_id:
            material_df.index = material_df.type
            ids = material_df.loc['Песок', 'id']                        ##!!!! Временный материал для кривых входящих данных
            layers_id.append(ids)

        if license_name  == '':
            license_name = item['license']
        if river == '':
            river = item['river']
        if  slice_name == '':
            slice_name = item['slice_number']

        #Вызовем функцию проверки координат
        input_data_tester(gps, borehole_number)
        
        #Воостановим вид датафрейма материалов
        material_df.reset_index(drop= True , inplace= True)
        material_df   

        
        row = {
            'number':float(borehole_number),
            'x':gps[0],
            'y':gps[1],
            'z':gps[2],
            'layers_power':layers_power,
            'num_of_layers':len(layers_power),
            'layers_id':layers_id
        }

        global slice_data
        slice_data=slice_data.append(row, ignore_index=True)


#конвертер строкового номера скважины в числовой
def number_converter(string_num):
    
    replacements = {'0': '-', 'К': '0.01', 'А': '0.02', 'Б': '0.03'}     #делаем заменители символов для допскважин
    
    new_elem = ''
    for sym in string_num:
        if sym == '0':
            sym = '-'
        elif sym == 'К':
            sym = '.01'
        elif sym == 'А':
            sym = '.02'
        elif sym == 'Б':
            sym = '.03'
        
        new_elem=new_elem+sym

    return new_elem

#проверка на наличие координат
def input_data_tester(data, num):
    global file_adress
    global error
    
    messages = [
        ' Отсутствуют координата ',
        ' Отсутствуют координаты скважины ',
        ' Проверьте координату ',
        ' Проверьте координаты скважины ', 
    ]
    
    coord = ['X','Y','Z']
    error_text = 'Ошибка данных скважины '+ str(num) + '\n\n'
    status = ''
    height_up = .2
    
    for i, elem in enumerate(data):
    
        try:
            1 / float(elem)
        except:
            status = status + messages[2] + coord[i] + '\n'
            error = 1
            height_up = height_up + .1
            

    if error != 0:
        error_text = error_text + str(status)

        # делаем рамку
        left, width = .05, 1.
        bottom, height = .55, height_up
        right = left + width
        top = bottom + height

        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])

        # axes coordinates: (0, 0) is bottom left and (1, 1) is upper right
        p = patches.Rectangle(
        (left, bottom), width, height,
        fill=False, transform=ax.transAxes, clip_on=False
        )

        ax.add_patch(p)

        ax.text(0.5*(left+right), 0.5*(bottom+top), error_text,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=20, color='red',
            transform=ax.transAxes)

        ax.set_axis_off()
        #plt.show() 
        
        #записываем изображение по собранному адресу
        addr = str(save_fig(1))
        file_adress = addr
        plt.savefig(addr, format="svg")
        

    
   
    
    
def save_fig(error=None):
    #сохраняем изображение как svg
    #проверяем, есть ли индикатор ошибки и формируем начало названия файла
    
    image_dir = '/home/maks/proj/geology_backend/geology_proj/media/' + license_name + '/' + river + '/' + slice_name         #/home/maks/proj/{license}
    #print(image_dir,'//', image_name, 'сюда закинем')

    if os.path.exists(image_dir) == False:
        os.makedirs(image_dir)
        #print('Нет такой папки, создаём!')

    if error == 1:
        image_name = 'Ошибка'
        image_adress = (image_dir +'/'+ image_name + '.svg')
    
        return image_adress

    else:    
        image_name = 'Черновой разрез ' + str(slice_name)           #+ '.svg'


    if os.path.isfile(image_dir + '/' + image_name + '.svg') == True:
        files = os.listdir(image_dir)
        #image_files = filter(lambda x: x.endswitch('.svg'), files)
        #print(image_files, 'image_files')
        #print('а файл есть уже', image_name)
        #print('файлы тут', files)
        image_ver_counter = 2
        ver_list = []
        for file in files:
            #file = file[0:1]
            inin = len(image_name + ' Вер ')
            oout = len(file)-4
            file2 = file[inin:oout]
            #print(file2, 'file2')
            if len(file2) > 0:
                ver_list.append(int(file2))
                if image_ver_counter <= max(ver_list):
                    image_ver_counter = max(ver_list)+1

        #print(image_ver_counter)
            
            
        image_name = image_name + ' Вер ' + str(image_ver_counter)
    

    image_adress = (image_dir +'/'+ image_name + '.svg')
    #print(image_adress, '=> image_adress')

    return image_adress

#################################################

def main(list_of_boreholes_dict):
    global file_adress
    global slice_data
    global types_df
    global project_layers

    #примем список словарей снаружи
    data_extractor(list_of_boreholes_dict)

    if error == 1:
        print(file_adress, 'if работает')
        return file_adress

    print('if не работает')
    short_elem = list(material_df['short'])


    for i in range(0,len(slice_data)):
        types_agregator(i)


    #Если один из типов материала отсутствует, например нет рудного пласта, до убираем этот тип из списка обработки
    types_df.dropna(axis = 1, how ='all', inplace = True)
    types = list(types_df.columns)



    #с помощью списка имеющихся типов создаём последователный список слоёв в разрезе
    #и меняем краткий список типов на полный
    types_df = types_df.fillna(6)

    project_layers = []    
    #standart_group_list = ['prs', 'torfa', 'rud_plast', 'plotik']

    for elem in types:
        project_layers.append(one_type_layer_calc3(elem))                     #используем вер 3

    project_layers = (list(chain.from_iterable(project_layers)))    


    types = []
    for elem in project_layers:
        types.append((material_df['category'][elem]))


    #собираем данные о отметке и глубине реальных скважин для таблицы
    true_z_raw = list(slice_data['z'])
    true_z = []
    for el in true_z_raw:
        true_z.append(round((el),1))

    deep_raw = list(slice_data['layers_power'])
    deep = []
    for el in deep_raw:
        deep.append(sum(el))
        

    #используя генератор слоев создаём для каждой скважины полный набор слоёв включая нулевые
    slice_data['full_layers'] = slice_data.apply(lambda x: layer_generator(x['layers_power'], x['layers_id'], project_layers), axis =  1)  


    #вычисляем расстояние между скважинами и выводим количество шагов
    step, table_step = step_calculator(slice_data['x'], slice_data['y'])
    layers = layers_power_generator()

    #копируем расстояние между скважинами для построения таблицы
    table_step = table_step[1:]


    true_t_s = table_step.copy()
    true_t_s.append(0) #заполнитель последнего столбца таблицы иначе получается несоответствие длинны и таблица не формируется


    #создаём список номеров скважин, он нам пригодится для выделения реальных скважин
    boreholes_list = []
    true_bh_list = []
    for bh_item in range(0, slice_data.shape[0]):
        boreholes_list.append(float(slice_data['number'][bh_item]))
        true_bh_list.append(round(float(slice_data['number'][bh_item])))
    #print(boreholes_list, true_bh_list)
                            

    slice_data_true = slice_data.copy(deep=True)


    #применяем функцию генерации виртуальных скважин к данным разреза
    for bhole in range(0, slice_data.shape[0]-1):
        transition_detector(bhole, types)
        #virtual_bh_generator(bhole)
        #print('Обработана скважина: ', bhole, '\n###\n')



    #делаем сортировку по номеру скважины, учитывая виртуальные и обновляем индексы строк, тк они сбиваются после сортировки
    slice_data = slice_data.sort_values(by='number')
    slice_data.reset_index(drop= True, inplace= True)


    #считаем сумму толщин слоёв
    summ_slice_power = []

    for data in slice_data['full_layers']:
        summ_slice_power.append(sum(data))
        
    #print(len(summ_slice_power))
    #считаем сумму нулевой уровень (виртуальный слой от 0 до фактического уровня, от которого известен состав и будем складывать стек слоёв вверх

    zero_level=[]
    for i in range(0,slice_data.shape[0]):
        #print(slice_data.iloc[i][3])
        zero_level.append(slice_data.iloc[i][5]-summ_slice_power[i]) #нах тут -2
        

    #zero_level
    step, nodata = step_calculator(slice_data['x'], slice_data['y'])
    layers = layers_power_generator()


    #используя список реальных скважин, получаем их индексы в полной таблице, это нам пригодится для разметки
    true_bh_raw = []
    deep_raw = []
    for bh in boreholes_list:

        true_bh_raw.append(slice_data.index[slice_data['number'] == bh].tolist())
            
    true_bh = sum(true_bh_raw, [])  #финт, где мы избавляемся от вложенного списка из одного объекта выводя его как сумму  


    #здесь ищем коэфициент, чтоб связать длинну таблицы с длинной разреза
    for i, elem in enumerate(table_step):
        table_step[i] = elem *0.0053 #0.00271коеффициент соответствия шага скважин и таблицы


    #добавим хвосты в начало и конец списка zero_level и задерём их на 1м
    zero_level.append(zero_level[-1])
    zero_level.insert(0, zero_level[0])

    zero_level[0] = zero_level[0]+1
    zero_level[-1] = zero_level[-1]+1


    #добавим хвосты в начало и конец списка step и сдвинем их на 5м+_ от начала и конца списка шагов
    step.append(step[-1]+5)
    step.insert(0, -5)


    #добавим в таблицу слоёв соответствующие данные про хвосты
    #layers0 = layers.append(layers

    start_layers = layers.iloc[[0]]#.tolist()

    end_layers = layers.iloc[[-1]]#.tolist()


    res = pd.concat([start_layers, layers], ignore_index=True)
    res = pd.concat([res, end_layers], ignore_index=True)
    layers = res


    #генерим список тире для заполнения пост заполнения полей
    tire_symbol = [] 
    for i in range(0, len(true_bh_list)):
        tire_symbol.append('-')


    #описание таблицы
    #длинна table_step меньше на 1
    columns = true_bh


    torfa_power = tire_symbol
    sands_power = tire_symbol
    au_in_sands_mgm3 = tire_symbol
    mass_power_m = tire_symbol
    au_in_mass_mgm3 = tire_symbol

    row = ('Номер скважины',
        'Расстояние м/у скважинами',
        'Абс. отметка устья скважины',
        'Глубина выработки, м',
        'Мощность торфов, м',
        'Мощность песков, м',
        'Содержание золота в песках, мг/м3',
        'Мощность массы, м',
        'Содержание золота в массе, мг/м3'
        )

    cell_text = [true_bh_list,
                true_t_s,
                true_z,
                deep,
                torfa_power,
                sands_power,
                au_in_sands_mgm3,
                mass_power_m,
                au_in_mass_mgm3
                
                ]


    #print(len(cell_text), len(row), len(true_bh))

    table_step.append(table_step[len(table_step)-1]) #пока костыль для дополнительного столбца, чтоб всё совпало) где-то лишние данные

    new_tablestep = []

    start=0

    for i, item in enumerate(table_step):
        #print(i, item)
        next_step = round((start/2 + item/2), 4)
        start = item
        #print('start', start, next_step)
        
        new_tablestep.append(next_step)
        
        

    ####отображение графика

    #масштаб графика
    #A4 == 8,3 × 11,7 в дюймах
    x_size_fig = 11.7/1.2 #попытка подогнать под масштаб А4, но всё равно уезжает
    y_size_fig = 8.3/1.2


    #вызываем и задаём размер
    fig, ax = plt.subplots()
    fig.set_size_inches(x_size_fig, y_size_fig) 

    ratio = 5 #пропорция масштабов x, y
    ax.set_aspect(ratio)



    #легенда

    titl = 'Схематический геологический разрез '+str(slice_name) + '\n'+ 'Долина ' + str(river)
    ax.set_title(titl, pad = 60).set_fontsize(12)

    ax.text(10, slice_data['z'].max()+4,
            'Масштаб: горизонтальный - 1:1000\n                  вертикальный - 1:200\n Азимут:', fontsize = 8)




    #задаём лимиты осей
    plt.xlim(min(step)-10, max(step)+10) #вроде как не нужно, протестировать
    plt.ylim(slice_data['z'].min()-max(summ_slice_power)-4, slice_data['z'].max()+2)
    plt.xticks([])  #убираем разметку по Х
    plt.yticks(
        np.arange(round(slice_data['z'].min()-max(summ_slice_power)-2),
        slice_data['z'].max()+2, step=2)
        )# здесь мы задали шаг 2 метра для оси y



    # Прозрачные рамки 
    plt.gca().spines["top"].set_alpha(0)
    plt.gca().spines["bottom"].set_alpha(0)
    plt.gca().spines["right"].set_alpha(0)
    plt.gca().spines["left"].set_alpha(1)



    #создаём стек из столбцов датасета, сложение снизу наверх, с 'нулевого уровня' который формирует основу,
    #на которую складываются остальные слои согласно геологическому строению
    stack = [zero_level]

    for item in range((len(project_layers)-1),0-1, -1):
        stack.append(list(layers[item]))
        

    y = np.vstack(stack)
        #[zero_level,  layers[9], layers[8], layers[7], layers[6], layers[5], layers[4],layers[3], layers[2], layers[1], layers[0]])


    #цвета линий
    color_map = ['#ffffff']
    for code in project_layers[::-1]:
        color_map.append(material_df['color'][code])
    #print(color_map)    


    #отрисовываем графики
    all_stack = ax.stackplot(step, y,
                labels=layers.keys(),colors=color_map, alpha=1, edgecolor = 'black', linewidth = 0.2) #baseline = 'zero'


    #отрисовываем реальные скважины
    for i in true_bh:
        ax.plot(
            [step[i+1], step[i+1]], [zero_level[i+1]+0.03, slice_data['z'][i]],
            label="Line", color='black', alpha=1,linewidth = 0.5
            ) 
        
        

    #отрисовка таблицы
    table = ax.table(
        cellText  = cell_text,
        colWidths = new_tablestep, 
        rowLabels = row,  
        colLabels = None, 
        cellLoc   = 'center',  
        loc       = 'bottom'
    )


    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)

    n=29
    an1 = ax.annotate(n, xy=(0.5, 0.5), xycoords="data", color='r',
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round", fc="w"))


    #записываем изображение по собранному адресу
    addr = str(save_fig())
    file_adress = addr
    plt.savefig(addr, format="svg")

    #выводим изображение
    #plt.tick_params(axis='both', which='major', labelsize=9)
    #plt.show()
    
    slice_data.drop([0, len(slice_data)]) #грохнем датафрейм
    return file_adress



main([{'slice_number': 'пл 250', 'river': 'р. Сарала', 'borehole_number': '3', 'gps': [2345.0, 546.0, 433.0], 'layers_power': [1.0, 3.0, 4.0], 'layers_id': ['ПРС', 'Торф', 'ОКВ по габбро'], 'license': 'Гидра'}, 
{'slice_number': 'пл 250', 'river': 'р. Сарала', 'borehole_number': '5', 'gps': [0, 345, 234.0], 'layers_power': [4.0, 5.0, 6.0, 2.0, 2.5], 'layers_id': ['Торф', 'ОКВ по габбро', 'ПРС', 'ПРС', 'ОКВ по габбро'], 'license': 'Гидра'}, 
{'slice_number': 'пл 250', 'river': 'р. Сарала', 'borehole_number': '6', 'gps': [546, 234, 456], 'layers_power': [3.0], 'layers_id': ['ОКВ по габбро'], 'license': 'Гидра'}]

) 
    