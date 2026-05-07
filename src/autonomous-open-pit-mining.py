# =====================================================================
# FILE: autonomous-open-pit-mining
# =====================================================================
import random
import sys
import os
import simpy
import pandas as pd
from desk.stats.factorial import FactorialExperiment
from desk.stats.replication import ReplicationFramework    
from desk.analytics.financial import FinancialAnalyzer
from desk.validation.resource_validator import ResourceValidator
from desk.core.simulation_model import SimulationModel
from desk.core.simulation_observer import SimulationObserver
from desk.core.model_variables import ModelVariableTracker
from desk.core.entity import EventLogger
from desk.blocks.create_block import CreateBlock
from desk.blocks.process_block import ProcessBlock, MultiProcessBlock
from desk.blocks.sync_process_block import SyncProcessBlock
from desk.blocks.decide_block import DecideBlock
from desk.blocks.dispose_block import DisposeBlock
from desk.analytics.metrics import MetricsCollector
from desk.analytics.reporting import SimulationReporter
from desk.analytics.plotting import SimulationPlotter
from desk.validation.stability import StabilityAnalyzer
from desk.validation.warmup import WarmUpAnalyzer
from desk.config.simulation_config import SimulationConfig
from desk.visualization.interface import run_visualization


# ####################################################################################
# Projeto: Uso de caminhões autônomos em mina a céu aberto
# Autor: Lúcio Passos <lucio.passos@gmail.com>
# Descrição: Este projeto de simulação a eventos discretos investiga uma mina a céu aberto
# numa configuração 4x4 (quatro áreas de extração e quatro áreas de britagem) conectados em rede
# por uma frota de caminhões. Diversas métricas são avaliadas para quatro cenários:
# 1) 26 caminhões convencionais
# 2) 26 caminhões auotônomos
# 3) 15 caminhões convencionais
# 4) 15 caminhões auotônomos

FROTA = 15 #15/26
AUTONOMO = True  #True/False
# #######################################################################################

MTBFCORRECTION = 1.18 if (FROTA == 15) else 1   # este ajuste é necessário, pois com menos caminhões
                                                # a função mtbf_operador_adaptativo() tem menos amostras
                                                # pra conseguir garantir um mtbf no alvo. O ajuste garante
                                                # o mesmo mtbf para 15 e para 26 caminhões convencionais. 
CUSTOM_REPLICATION_KPIS = []

def calcula_kpis_frota(model, seed=None):
    tracker = model.variable_tracker

    def div0(a, b):
        return a / b if b else 0

    sim_time = model.env.now

    carregamento_FL = model.blocks["Loading"]
    recupera_brita_transp_BS1 = model.blocks["Crushing"]

    input_minerio = tracker.get_final("input_minerio")
    input_esteril = tracker.get_final("input_esteril")
    input_total = input_minerio + input_esteril

    proc_minerio = (
        tracker.get_final("proc_BS1_minerio")
    )

    proc_esteril = (
        tracker.get_final("proc_BS1_esteril")
    )

    proc_total = proc_minerio + proc_esteril

    falhas = tracker.get_final("kpi_falhas_caminhao")
    tempo_parado = tracker.get_final("kpi_tempo_parado_caminhao")

    mtbf_real = div0(
        tracker.get_final("kpi_soma_mtbf_real"),
        tracker.get_final("kpi_n_mtbf_real")
    )

    mttr_real = div0(
        tracker.get_final("kpi_soma_mttr_real"),
        tracker.get_final("kpi_n_mttr_real")
    )

    tempo_medio_carregando = div0(
        tracker.get_final("kpi_soma_tempo_carregando"),
        tracker.get_final("kpi_n_tempo_carregando")
    )

    tempo_medio_cheio = div0(
        tracker.get_final("kpi_soma_tempo_cheio"),
        tracker.get_final("kpi_n_tempo_cheio")
    )

    tempo_medio_descarregando = div0(
        tracker.get_final("kpi_soma_tempo_descarregando"),
        tracker.get_final("kpi_n_tempo_descarregando")
    )

    tempo_medio_fila_carregamento = div0(
        carregamento_FL.total_queue_time,
        carregamento_FL.entities_processed
    )

    tempo_medio_fila_descarregamento = div0(
        recupera_brita_transp_BS1.total_queue_time,
        recupera_brita_transp_BS1.entities_processed
    )

    tempo_medio_vazio = div0(
        tracker.get_final("kpi_soma_tempo_vazio"),
        tracker.get_final("kpi_n_tempo_vazio")
    )



    # =========================
    # TEMPOS BÁSICOS
    # =========================
    tempo_total_frota = FROTA * sim_time

    tempo_parado = tracker.get_final("kpi_tempo_parado_caminhao")

    tempo_disponivel_frota = tempo_total_frota - tempo_parado

    tempo_ativo_frota = (
        tracker.get_final("kpi_soma_tempo_carregando")
        + tracker.get_final("kpi_soma_tempo_cheio")
        + tracker.get_final("kpi_soma_tempo_vazio")   # <-- NOVO
        + tracker.get_final("kpi_soma_tempo_descarregando")
    )

    # =========================
    # KPI's FROTA
    # =========================
    disponibilidade = 100 * div0(
        tempo_disponivel_frota,
        tempo_total_frota
    )

    utilizacao_calendario = 100 * div0(
        tempo_ativo_frota,
        tempo_total_frota
    )

    utilizacao_sobre_disponivel = 100 * div0(
        tempo_ativo_frota,
        tempo_disponivel_frota
    )


    taxa_carregamento = div0(
        carregamento_FL.entities_processed,
        sim_time
    )

    numero_medio_caminhoes_fila_carregamento = (
        taxa_carregamento * tempo_medio_fila_carregamento
    )

    tempo_ocioso_disponivel = tempo_disponivel_frota - tempo_ativo_frota

    ociosidade_sobre_disponivel = 100 * div0(
        tempo_ocioso_disponivel,
        tempo_disponivel_frota
    )

    return {
        "seed": seed,
        "total_entidades_criadas": model.entity_count,

        "input_minerio": input_minerio,
        "input_esteril": input_esteril,
        "entrada_minerio_pct": 100 * div0(input_minerio, input_total),
        "entrada_esteril_pct": 100 * div0(input_esteril, input_total),

        "processado_minerio_ent": proc_minerio,
        "processado_esteril_ent": proc_esteril,
        "processado_minerio_pct": 100 * div0(proc_minerio, proc_total),
        "processado_esteril_pct": 100 * div0(proc_esteril, proc_total),

        "numero_total_falhas": falhas,
        "tempo_total_simulacao_min": sim_time,
        "tempo_total_simulacao_h": sim_time / 60,
        "tempo_total_simulacao_dias": sim_time / 1440,

        "disponibilidade_media_frota_pct": disponibilidade,
        "tempo_ocioso_disponivel_frota_pct": tempo_ocioso_disponivel,
        "ociosidade_sobre_disponivel_frota_pct": ociosidade_sobre_disponivel,
        "utilizacao_sobre_disponivel_media_frota_pct": utilizacao_sobre_disponivel,
        "utilizacao_calendario_media_frota_pct": utilizacao_calendario,
        "n_amostras_mtbf_real": tracker.get_final("kpi_n_mtbf_real"),
        "mtbf_medio_real_min": mtbf_real,
        "mttr_medio_real_min": mttr_real,

        "ton_minerio_processadas": tracker.get_final("kpi_ton_minerio"),
        "ton_esteril_processadas": tracker.get_final("kpi_ton_esteril"),

        "tempo_medio_carregando_min": tempo_medio_carregando,
        "tempo_medio_deslocando_cheio_min": tempo_medio_cheio,
        "tempo_medio_descarregando_min": tempo_medio_descarregando,
        "tempo_medio_deslocando_vazio_min": tempo_medio_vazio,

        "tempo_medio_fila_carregamento_min": tempo_medio_fila_carregamento,
        "tempo_medio_rom_aguardando_britagem_min": tempo_medio_fila_descarregamento,
        "numero_medio_caminhoes_fila_carregamento": numero_medio_caminhoes_fila_carregamento,

        "entidades_carregadas": carregamento_FL.entities_processed,
        "entidades_britadas_processadas": recupera_brita_transp_BS1.entities_processed,
        "tempo_medio_aguardando_britagem_min": tempo_medio_fila_descarregamento,
    }

# ================================================================
# Each ACD model is implemented here
# ================================================================
def build_model(final_simulation_time=None, event_logger=None, verbose=False,
                        entity_filter=None, resource_filter=None,
                        event_type_filter=None, time_range=None): 
    
    HOURS = 60  # Time conversion factor (base time: minutes)
    DAYS = 1440
    YEARS = 525600

    MTBF_MINIMO = 1.0

    if final_simulation_time is None:
        final_simulation_time = 365 * DAYS  # Set default to match the intended simulation time
    
    model = SimulationModel(verbose=verbose,
        entity_filter=entity_filter,
        resource_filter=resource_filter,
        event_type_filter=event_type_filter,
        time_range=time_range)  # NEW: Pass verbose flag

    # após criada uma entidade ROM, seleciona aleatoriamente o atributo (minério/estéril) 
    # na proporção 0.794 para minério 
    def escolhe_material_extracao(p=0.794, memoria=0):
        if not hasattr(model, "last_input_material_bin"):
            model.last_input_material_bin = None

        if not (0.0 <= p <= 1.0):
            raise ValueError(f"'p' deve estar entre 0 e 1. Recebido: {p}")
        if not (0.0 <= memoria <= 1.0):
            raise ValueError(f"'memoria' deve estar entre 0 e 1. Recebido: {memoria}")

        last = model.last_input_material_bin

        if last is None:
            x = 1 if random.random() < p else 0
        else:
            prob_1 = p + memoria * (1.0 - p) if last == 1 else (1.0 - memoria) * p
            x = 1 if random.random() < prob_1 else 0

        model.last_input_material_bin = x
        material = "minerio" if x == 1 else "esteril"

        if hasattr(model, "variable_tracker"):
            if material == "minerio":
                atual = model.variable_tracker.get_current("input_minerio")
                model.variable_tracker.update("input_minerio", model.env.now, atual + 1)
            else:
                atual = model.variable_tracker.get_current("input_esteril")
                model.variable_tracker.update("input_esteril", model.env.now, atual + 1)

        return material

    # caracteriza cada entidade caminhao com um codigo de CA101 até CA126 
    def gera_codigo_caminhao():
        if not hasattr(model, "contador_caminhao"):
            model.contador_caminhao = 101

        codigo = f"CA{model.contador_caminhao}"
        model.contador_caminhao += 1
        return codigo

    # as funções abaixo atribuem a cada caminhao um timestamp de inicio de operação, 
    # que pode ser após ser criado ou após retornar de uma parada, e um timestamp com o instante 
    # prevista para uma parada de acordo com a curva do mtbf 
    def gera_tempo_inicio_operacao():
        return model.env.now
    def gera_tempo_prox_parada():
        #return model.env.now + distribution("mtbf_operador")
        return model.env.now + mtbf_operador_adaptativo()

    # ajuste no mtbf. Como a checagem de tempo de operação só ocorre na viagem de retorno do caminhão
    # então é natural que o mtbf real fique dilatado pelo tempo de ciclo, então esta função
    # vai adaptando o mtbf solicitado dos próximos caminhões de maneira que a média geral do mtbf fique
    # próximo do real  
    def mtbf_operador_adaptativo():
        mtbf_base = distribution("mtbf_operador")

        atraso = getattr(model, "ultimo_atraso_parada_operador", 0.0)

        mtbf_corrigido = mtbf_base - atraso

        return max(MTBF_MINIMO, mtbf_corrigido)

    #as funções abaixo são apenas para registros de relatório
    def registra_entrada_falha_operador(entity, block_name, route_taken, time):
        if entity.get_attribute("entity_type") != "Caminhao":
            return

        tracker.update(
            "kpi_falhas_caminhao",
            time,
            tracker.get_current("kpi_falhas_caminhao") + 1
        )

        # MTBF real: do fim do reparo anterior até a nova falha.
        # Desconsidera o primeiro tempo desde a criação.
        ultimo_fim_reparo = entity.get_attribute("ultimo_fim_reparo_operador", None)

        if ultimo_fim_reparo is not None:
            intervalo = time - ultimo_fim_reparo

            tracker.update(
                "kpi_soma_mtbf_real",
                time,
                tracker.get_current("kpi_soma_mtbf_real") + intervalo
            )

            tracker.update(
                "kpi_n_mtbf_real",
                time,
                tracker.get_current("kpi_n_mtbf_real") + 1
            )

        tempo_prox_parada = entity.get_attribute("tempo_prox_parada", time)
        atraso = max(0.0, time - tempo_prox_parada)

        model.ultimo_atraso_parada_operador = atraso

        entity.add_attribute("inicio_parada_operador", time)
    def registra_fim_reparo_operador(entity, block_name, service_time, time):
        if entity.get_attribute("entity_type") != "Caminhao":
            return

        tracker.update(
            "kpi_tempo_parado_caminhao",
            time,
            tracker.get_current("kpi_tempo_parado_caminhao") + service_time
        )

        tracker.update(
            "kpi_soma_mttr_real",
            time,
            tracker.get_current("kpi_soma_mttr_real") + service_time
        )

        tracker.update(
            "kpi_n_mttr_real",
            time,
            tracker.get_current("kpi_n_mttr_real") + 1
        )

        entity.add_attribute("ultimo_fim_reparo_operador", time)
    def registra_dispose_minerio(entity, block_name, time):
        if entity.get_attribute("entity_type") != "ROM":
            return

        tracker.update(
            "kpi_ton_minerio",
            time,
            tracker.get_current("kpi_ton_minerio") + TON_POR_ENTIDADE
        )

        registra_tempos_ciclo_rom(entity, time)
    def registra_dispose_esteril(entity, block_name, time):
        if entity.get_attribute("entity_type") != "ROM":
            return

        tracker.update(
            "kpi_ton_esteril",
            time,
            tracker.get_current("kpi_ton_esteril") + TON_POR_ENTIDADE
        )

        registra_tempos_ciclo_rom(entity, time)
    def registra_tempos_ciclo_rom(entity, time):
        tempo_carregando = entity.get_attribute("Loading_service_time", None)
        tempo_cheio = entity.get_attribute("Delay BS1_service_time", None)
        tempo_descarregando = entity.get_attribute("Crushing_service_time", None)

        if tempo_carregando is not None:
            tracker.update(
                "kpi_soma_tempo_carregando",
                time,
                tracker.get_current("kpi_soma_tempo_carregando") + tempo_carregando
            )
            tracker.update(
                "kpi_n_tempo_carregando",
                time,
                tracker.get_current("kpi_n_tempo_carregando") + 1
            )

        if tempo_cheio is not None:
            tracker.update(
                "kpi_soma_tempo_cheio",
                time,
                tracker.get_current("kpi_soma_tempo_cheio") + tempo_cheio
            )
            tracker.update(
                "kpi_n_tempo_cheio",
                time,
                tracker.get_current("kpi_n_tempo_cheio") + 1
            )

        if tempo_descarregando is not None:
            tracker.update(
                "kpi_soma_tempo_descarregando",
                time,
                tracker.get_current("kpi_soma_tempo_descarregando") + tempo_descarregando
            )
            tracker.update(
                "kpi_n_tempo_descarregando",
                time,
                tracker.get_current("kpi_n_tempo_descarregando") + 1
            )
    def registra_tempo_vazio(entity, block_name, service_time, time):
        if entity.get_attribute("entity_type") != "Caminhao":
            return

        tracker.update(
            "kpi_soma_tempo_vazio",
            time,
            tracker.get_current("kpi_soma_tempo_vazio") + service_time
        )

        tracker.update(
            "kpi_n_tempo_vazio",
            time,
            tracker.get_current("kpi_n_tempo_vazio") + 1
        )

# essas distribuições abaixo podem ser usadas futuramente em uma investifação a respeito de microparadas
#            'mtbf_maquinario_1': random.erlangvariate(3332.745744, 2,0.000220),
#            'mtbf_maquinario_1': (0.000220 + random.gammavariate(2, 3332.745744)),
#            'mtbf_maquinario_2': (random.triangular(5936.982192, 142069.888224002, 23345.28288)),
#            'mtbf_maquinario_3': (0.283319 + random.gammavariate(0.284372, 5856.492465)),
#            'mtbf_BS1_microparada': (159.33 + (27613.551 - 159.33) * random.betavariate(0.5652, 4.2786)),
#            'mtbf_maquinario_5': (48550.7339691598 + random.expovariate(1 / 152097.288252936)),
#            'mtbf_maquinario_6': (random.weibullvariate(430.672129, 1.184934)),
#            'mtbf_maquinario_7': (258.650 + random.weibullvariate(1.071, 1039.835)),
#            'mttr_maquinario_1': (0.004402 + random.gammavariate(4.013445, 19.079194)),
#            'mttr_maquinario_2': (random.triangular(199.68336, 391.5, 302.7168)),
#            'mttr_maquinario_3': (2.57 + (70.803 - 2.57) * random.betavariate(0.826, 5.722)),
#            'mttr_BS1_microparada': (64.956049 + random.gammavariate(1.513766, 36.548794)),
#            'mttr_maquinario_5': (369.970190319399 + random.expovariate(1 / 309.107601680601)),
#            'mttr_maquinario_6': (0.249987 + random.weibullvariate(9.456887, 2.385616)),
#            'mttr_maquinario_7': (18.78 + random.weibullvariate(0.955, 7.879)),

    def distribution(tipo):
    # Unidade básica para todos os tempos: minutos
        dist = {
            #----Frente de Lavra---------------
            'chegadaCaminhoes': lambda: random.uniform(0, 0),
            'chegadaROM': lambda: random.uniform(1.64, 1.64),

            'carregamento_FL': lambda: (-8.43323 + 248.276 * random.betavariate(77.5073, 1506.59)) * 1.5,
            'mtbf_FL': lambda: 0.016666 + 496.441 * random.betavariate(0.503515, 4.53269),
            'mttr_FL': lambda: random.gammavariate(0.486516, 44.2687),

            #----Britagens----------
            'delay_FL_BS1': lambda: 1 + (-0.270587) + 3.90734e+06 * random.betavariate(3.61331, 2.34131e+06),
            'delay_BS1_FL': lambda: -0.270587 + 3.90734e+06 * random.betavariate(3.61331, 2.34131e+06),

            'recupera_brita_transp_BS1': lambda: random.uniform(0.76, 0.76),

            'mtbf_BS1_microparada': lambda: 159.33 + (27613.551 - 159.33) * random.betavariate(0.5652, 4.2786),
            'mttr_BS1_microparada': lambda: 64.956049 + random.gammavariate(1.513766, 36.548794),
        }

        #----Caminhão----------
        if AUTONOMO:
            dist.update({
                'mtbf_operador': lambda: (random.gammavariate(0.78299, 79.411))*MTBFCORRECTION,
                'mttr_operador': lambda: random.weibullvariate(9.93131, 0.942199),
            })
        else:
            dist.update({
                'mtbf_operador': lambda: (random.gammavariate(0.683972, 52.0876))*MTBFCORRECTION,
                'mttr_operador': lambda: random.expovariate(0.082185),
            })

        return dist.get(tipo, lambda: 0.0)()

    #============================== RECURSOS =========================================
    # Resources - regular, priority, preempt
    
    escavadeira_FL = model.add_resource("escavadeira_FL", 4, "preemptive")
    model.set_resource_reliability(
        "escavadeira_FL",
        time_to_failure_fn=lambda: distribution('mtbf_FL'),
        repair_time_fn=lambda: distribution('mttr_FL')
    )

    # o tamanho máximo do estoque do britador foi obtido somando a capacidade das 4 britagens e dividindo por 299,4
    MAX_QUEUE_BS1 = 16961

    BS1_maquinario_1 = model.add_resource("BS1_maquinario_1", 1, "preemptive", max_queue=MAX_QUEUE_BS1)
    model.set_resource_reliability(
        "BS1_maquinario_1",
        time_to_failure_fn=lambda: distribution('mtbf_BS1_microparada'),
        repair_time_fn=lambda: distribution('mttr_BS1_microparada')
    )    
    ''' # esses recursos abaixo podem ser usadas futuramente em uma investifação a respeito de microparadas
    BS1_maquinario_2 = model.add_resource("BS1_maquinario_2", 1, "preemptive", max_queue=MAX_QUEUE_BS1)
    model.set_resource_reliability(
        "BS1_maquinario_2",
        time_to_failure_fn=lambda: distribution('mtbf_maquinario_2'),
        repair_time_fn=lambda: distribution('mttr_maquinario_2')
    )    

    BS1_maquinario_3 = model.add_resource("BS1_maquinario_3", 1, "preemptive", max_queue=MAX_QUEUE_BS1)
    model.set_resource_reliability(
        "BS1_maquinario_3",
        time_to_failure_fn=lambda: distribution('mtbf_maquinario_3'),
        repair_time_fn=lambda: distribution('mttr_maquinario_3')
    )    

    BS1_maquinario_4 = model.add_resource("BS1_maquinario_4", 1, "preemptive", max_queue=MAX_QUEUE_BS1)
    model.set_resource_reliability(
        "BS1_maquinario_4",
        time_to_failure_fn=lambda: distribution('mtbf_maquinario_4'),
        repair_time_fn=lambda: distribution('mttr_maquinario_4')
    )    

    BS1_maquinario_5 = model.add_resource("BS1_maquinario_5", 1, "preemptive", max_queue=MAX_QUEUE_BS1)
    model.set_resource_reliability(
        "BS1_maquinario_5",
        time_to_failure_fn=lambda: distribution('mtbf_maquinario_5'),
        repair_time_fn=lambda: distribution('mttr_maquinario_5')
    )    

    BS1_maquinario_6 = model.add_resource("BS1_maquinario_6", 1, "preemptive", max_queue=MAX_QUEUE_BS1)
    model.set_resource_reliability(
        "BS1_maquinario_6",
        time_to_failure_fn=lambda: distribution('mtbf_maquinario_6'),
        repair_time_fn=lambda: distribution('mttr_maquinario_6')
    )    

    BS1_maquinario_7 = model.add_resource("BS1_maquinario_7", 1, "preemptive", max_queue=MAX_QUEUE_BS1)
    model.set_resource_reliability(
        "BS1_maquinario_7",
        time_to_failure_fn=lambda: distribution('mtbf_maquinario_7'),
        repair_time_fn=lambda: distribution('mttr_maquinario_7')
    )    
    '''
    model._start_resource_reliability_if_needed()
    
    #============================== BLOCOS =========================================
    # Create blocks -----------------------------------------------------------------------
    chegada_caminhoes = CreateBlock(
        "Truck Arrival", model.env,
        inter_arrival_time=lambda: distribution('chegadaCaminhoes'),
        entity_prefix="Caminhao",
        max_arrivals=FROTA,
        first_creation=0.0,
        # priority_generator=prio("Cliente"),
        event_logger=event_logger
    )
    chegada_rom = CreateBlock(
        # 10.970 t/h, entidade = 300 t
        # Taxa: 10970 / 300 = 36,57 entidades por hora
        # Convertendo para minutos: 60 / 36,57 = 1,64 min por entidade
        # inter_arrival_time=lambda: 1.64
        "ROM Arrival", model.env,
        inter_arrival_time=lambda: distribution('chegadaROM'),
        entity_prefix="ROM",
        max_arrivals=10000000, # Infinito
        first_creation=0.0,
        event_logger=event_logger
    ) 

    chegada_caminhoes.assign_attributes(
        entity_type="Caminhao",
        codigo_caminhao=gera_codigo_caminhao,
        tempo_inicio_operacao=gera_tempo_inicio_operacao,
        tempo_prox_parada=gera_tempo_prox_parada
    )

    chegada_rom.assign_attributes(entity_type="ROM")

    # Decide blocks ----------------------------------------------------------------------- 
    split_retorno_caminhao = DecideBlock("Ret.Truck", model.env, decision_type="condition")
    split_parada_operador = DecideBlock("Human Fail?", model.env, decision_type="condition")
    split_minerio_esteril = DecideBlock("Ore/Ste?", model.env, decision_type="condition")
    
    # Process blocks -----------------------------------------------------------------------
    #
    carregamento_FL = SyncProcessBlock(
        "Loading", model.env,
        required_entity_types={"Caminhao": 1, "ROM": 1},
        primary_entity_type="Caminhao",
        resource=escavadeira_FL,
        delay_time=lambda: distribution('carregamento_FL'),
        resource_units=1,
        event_logger=event_logger
    ) 
    carregamento_FL.set_resource_name("escavadeiras_FL")

    carregamento_FL.assign_group_attributes(
        material=lambda: escolhe_material_extracao()
    )

    delay_FL_BS1 = SyncProcessBlock(
        "Delay BS1", model.env,
        required_entity_types={"Caminhao": 1, "ROM": 1},
        primary_entity_type="Caminhao",
        resource=None,
        delay_time=lambda: distribution('delay_FL_BS1'),
        resource_units=1,
        event_logger=event_logger
    ) 

    delay_BS1_FL = ProcessBlock(
        "Delay FL", model.env,
        resource=None,
        delay_time=lambda: distribution('delay_BS1_FL'),
        resource_units=1,
        event_logger=event_logger
    ) 

    recupera_brita_transp_BS1 = ProcessBlock(
        "Crushing", model.env,
        resource=BS1_maquinario_1,
        delay_time=lambda: distribution('recupera_brita_transp_BS1'),
        event_logger=event_logger
    )
    recupera_brita_transp_BS1.set_resource_name("BS1_maquinario_1")

    delay_parada_operador = ProcessBlock(
        "Operator Delay", model.env,
        resource=None,
        delay_time=lambda: distribution('mttr_operador'),
        resource_units=1,
        event_logger=event_logger
    ) 

    delay_parada_operador.assign_attributes(
        tempo_inicio_operacao=lambda: model.env.now,
        tempo_prox_parada=gera_tempo_prox_parada
    )
    
    dispose_minerio = DisposeBlock(
        "Ore Disposal", 
        model.env, 
        event_logger=event_logger)

    dispose_esteril = DisposeBlock(
        "Sterile Disposal", 
        model.env, 
        event_logger=event_logger)    
    
    #============================== ADICIONA BLOCOS =========================================
    # Add blocks to model
    for block in [chegada_caminhoes, chegada_rom, carregamento_FL, recupera_brita_transp_BS1,
        split_retorno_caminhao, split_parada_operador, split_minerio_esteril,
        delay_FL_BS1, delay_BS1_FL, delay_parada_operador,
        dispose_minerio, dispose_esteril]:
        model.add_block(block)

    #============================== CONECTA BLOCOS =========================================
    # Connect flow
    chegada_caminhoes.connect_to(carregamento_FL)
    chegada_rom.connect_to(carregamento_FL)                
    carregamento_FL.connect_to(delay_FL_BS1)
    delay_FL_BS1.connect_to(split_retorno_caminhao) 
    delay_BS1_FL.connect_to(split_parada_operador)
    delay_parada_operador.connect_to(carregamento_FL) 
    recupera_brita_transp_BS1.connect_to(split_minerio_esteril) 

    #============================== CONECTA INTERLOCKS =========================================
    BS1_maquinario_1.interlock_to(carregamento_FL)

    #============================== ROTEIA DECISÕES =========================================
    #----frente de lavra---

    #----Britagem na frente de lavra---
    split_retorno_caminhao.add_route(
        "retorno_caminhao",
        delay_BS1_FL,
        condition=lambda e: e.get_attribute("entity_type") == "Caminhao"
    )
    split_retorno_caminhao.add_route(
        "segue_ROM",
        recupera_brita_transp_BS1,
        condition=lambda e: e.get_attribute("entity_type") == "ROM"
    )
    
    split_parada_operador.add_route(
        "paradaOperador",
        delay_parada_operador,
        condition=lambda e: model.env.now >= e.get_attribute("tempo_prox_parada", float("inf"))
    )

    split_parada_operador.add_route(
        "segueSemParar",
        carregamento_FL,
        condition=lambda e: model.env.now < e.get_attribute("tempo_prox_parada", float("inf"))
    )

    split_minerio_esteril.add_route(
        "vaiParaTCLD",
        dispose_minerio,
        condition=lambda e: e.get_attribute("material", "") == "minerio"
    )
    split_minerio_esteril.add_route(
        "vaiParaOverland",
        dispose_esteril,
        condition=lambda e: e.get_attribute("material", "") == "esteril"
    )


    # ================================================================
    # CREATE OBSERVER (separate from blocks)
    # ================================================================   

    tracker = model.variable_tracker

    tracker.add_variable('input_minerio', 0, 'Total de entidades geradas como minério', 'entidades')
    tracker.add_variable('input_esteril', 0, 'Total de entidades geradas como estéril', 'entidades')

    tracker.add_variable('proc_BS1_minerio', 0, 'Minério processado no britador BS1', 'entidades')
    tracker.add_variable('proc_BS1_esteril', 0, 'Estéril processado no britador BS1', 'entidades')

    tracker.add_variable('proc_BS3_minerio', 0, 'Minério processado no britador BS3', 'entidades')
    tracker.add_variable('proc_BS3_esteril', 0, 'Estéril processado no britador BS3', 'entidades')

    tracker.add_variable('proc_BS2_minerio', 0, 'Minério processado no britador BS2', 'entidades')
    tracker.add_variable('proc_BS2_esteril', 0, 'Estéril processado no britador BS2', 'entidades')

    tracker.add_variable('proc_BS4_minerio', 0, 'Minério processado no britador BS4', 'entidades')
    tracker.add_variable('proc_BS4_esteril', 0, 'Estéril processado no britador BS4', 'entidades')

    observer = SimulationObserver(model)

    TON_POR_ENTIDADE = 300

    carregamento_FL = model.blocks["Loading"]
    recupera_brita_transp_BS1 = model.blocks["Crushing"]

    # =========================
    # KPI - variáveis brutas
    # =========================
    tracker.add_variable("kpi_falhas_caminhao", 0, "Número total de falhas da frota", "un")
    tracker.add_variable("kpi_tempo_parado_caminhao", 0, "Tempo total parado da frota", "min")

    tracker.add_variable("kpi_soma_mtbf_real", 0, "Soma dos intervalos reais entre reparo e próxima falha", "min")
    tracker.add_variable("kpi_n_mtbf_real", 0, "Número de intervalos reais de MTBF", "un")

    tracker.add_variable("kpi_soma_mttr_real", 0, "Soma dos tempos reais de reparo", "min")
    tracker.add_variable("kpi_n_mttr_real", 0, "Número de reparos", "un")

    tracker.add_variable("kpi_ton_minerio", 0, "Toneladas de minério descarregadas", "t")
    tracker.add_variable("kpi_ton_esteril", 0, "Toneladas de estéril descarregadas", "t")

    tracker.add_variable("kpi_soma_tempo_carregando", 0, "Soma dos tempos carregando", "min")
    tracker.add_variable("kpi_n_tempo_carregando", 0, "Número de carregamentos medidos", "un")

    tracker.add_variable("kpi_soma_tempo_cheio", 0, "Soma dos tempos deslocando cheio", "min")
    tracker.add_variable("kpi_n_tempo_cheio", 0, "Número de deslocamentos cheios medidos", "un")

    tracker.add_variable("kpi_soma_tempo_descarregando", 0, "Soma dos tempos descarregando", "min")
    tracker.add_variable("kpi_n_tempo_descarregando", 0, "Número de descarregamentos medidos", "un")

    tracker.add_variable("kpi_soma_tempo_vazio", 0, "Soma dos tempos deslocando vazio", "min")
    tracker.add_variable("kpi_n_tempo_vazio", 0, "Número de deslocamentos vazios", "un")

    def conta_material_entrada(entity, block_name, route_taken, time):
        if entity.get_attribute("entity_type", "") != "ROM":
            return

        material = entity.get_attribute("material", "")
        if material == "minerio":
            atual = tracker.get_current('input_minerio')
            tracker.update('input_minerio', time, atual + 1)
        elif material == "esteril":
            atual = tracker.get_current('input_esteril')
            tracker.update('input_esteril', time, atual + 1)

    def conta_BS1(entity, block_name, service_time, time):
        if entity.get_attribute("entity_type", "") != "ROM":
            return

        material = entity.get_attribute("material", "")
        if material == "minerio":
            atual = tracker.get_current('proc_BS1_minerio')
            tracker.update('proc_BS1_minerio', time, atual + 1)
        elif material == "esteril":
            atual = tracker.get_current('proc_BS1_esteril')
            tracker.update('proc_BS1_esteril', time, atual + 1)

    observer.on_entity_disposed(
        block_name="Ore Disposal",
        callback=registra_dispose_minerio
    )

    observer.on_entity_disposed(
        block_name="Sterile Disposal",
        callback=registra_dispose_esteril
    )

    observer.on_decision_made(
        block_name="Human Fail?",
        route_name="paradaOperador",
        callback=registra_entrada_falha_operador
    )

    observer.on_activity_complete(
        block_name="Operator Delay",
        callback=registra_fim_reparo_operador
    )

    observer.on_activity_complete(
        block_name="Crushing",
        callback=conta_BS1
    )  

    observer.on_activity_complete(
        block_name="Delay FL",
        callback=registra_tempo_vazio
    )

    def conta_material_entrada_loading(entity, block_name, service_time, time):
        if entity.get_attribute("entity_type", "") != "ROM":
            return

        material = entity.get_attribute("material", "")

        if material == "minerio":
            atual = tracker.get_current("input_minerio")
            tracker.update("input_minerio", time, atual + 1)

        elif material == "esteril":
            atual = tracker.get_current("input_esteril")
            tracker.update("input_esteril", time, atual + 1)

    #observer.on_activity_complete(
    #    block_name="Loading",
    #    callback=conta_material_entrada_loading
    #)
  
    return model


def simulation_wrapper(seed=None, until=None, warm_up_period=None):
    """Wrapper function for replication framework."""
    
    from desk.core.entity import EventLogger    
    event_logger = EventLogger()

    HOURS = 60  # Time conversion factor (base time: minutes)
    DAYS = 1440
    YEARS = 525600

    # Create configuration
    config = SimulationConfig(
        duration=20,
        warm_up_period=2,        
        seed= None ,#44,#123,
        check_stability=True
    )

    model = build_model(config.duration, event_logger, verbose=False)
   
    model.split_balance = {
        "minerio": {"BS3": 0, "BS2": 0},
        "esteril": {"BS3": 0, "BS2": 0},
    }

    model.run_simulation(
        validate_resources=False,
        until=until,
        seed=seed,
        warm_up_period=warm_up_period
    )
    
    CUSTOM_REPLICATION_KPIS.append(calcula_kpis_frota(model, seed=seed))

    return model

# ================================================================
# For full simulation
# ================================================================
# Run replications
def run_replications():
    global CUSTOM_REPLICATION_KPIS
    CUSTOM_REPLICATION_KPIS = []
    os.makedirs("results", exist_ok=True)
    replication_framework = ReplicationFramework(
        simulation_function=simulation_wrapper,
        n_replications=10
    )

    HOURS = 60  # Time conversion factor (base time: minutes)
    DAYS = 1440
    YEARS = 525600
    
    replication_framework.run_replications(
        base_seed=12345,
        until=31*DAYS,
        warm_up_period=1*DAYS
    )

    # Access results
    df = replication_framework.get_results_dataframe()

    if CUSTOM_REPLICATION_KPIS:
        df_relatorio = pd.DataFrame(CUSTOM_REPLICATION_KPIS)


        prefixo = f"{FROTA}_{'AUTO' if AUTONOMO else 'CONV'}_"

        nome_arquivo = f"results/{prefixo}relatorio_replicacoes_sbpo.csv"

        df_relatorio.to_csv(
            nome_arquivo,
            index=False,
            sep=";",
            decimal=","
        )

        print("\nRelatório das replicações:")
        print(df_relatorio.describe())

        print("\nArquivo gerado:")
        print("results/relatorio_replicacoes_sbpo.csv")

    print(df.describe())
# ================================================================
    
'''
# ================================================================
# Factorial Analysis
# ================================================================
def factorial_analysis():
    """Example of factorial analysis with hospital simulation."""

    HOURS = 60  # Time conversion factor (base time: minutes)
    DAYS = 1440
    YEARS = 525600

    # Create configuration
    config = SimulationConfig(
        duration=24*HOURS,
        warm_up_period=2*HOURS,        
        seed=123,
        check_stability=True        
    )
    
    # Define simulation function wrapper
    def simulation_wrapper(arrival_rate=15, num_troncos=30,
                                    seed=None, until=None, warm_up_period=0, **kwargs):
        """Wrapper that adapts parameters for factorial analysis."""

        # ############################################################
        # # O modelo de simulação é importado aqui
        # ############################################################
        
        # This would need to be modified in your actual model to accept these parameters
        # For now, this is a template showing how to structure it
        model = build_model(config.duration, verbose=False)
        model.run_simulation(validate_resources=False, until=until, seed=seed, warm_up_period=warm_up_period)
        return model
    
    
    # Create factorial analysis
    factorial = FactorialExperiment(
        simulation_function=simulation_wrapper,
        base_seed=12345
    )
    
    # Add factors
    factorial.add_factor(
        factor_name='arrival_rate',
        parameter_path='CreateBlock.inter_arrival_time',
        levels=[4/60, 8/60, 16/60],  # Minutes between arrivals
        description='Taxa de chegada de clientes (min)'
    )
    
    factorial.add_factor(
        factor_name='num_troncos',
        parameter_path='Resource.troncos.capacity',
        levels=[30, 31, 32],
        description='Número de troncos'
    )
    
    
    # Run experiment
    factorial.run_factorial_experiment(
        n_replications=5,
        simulation_time=40,  # 40 min
        warm_up_period=7,    # 7 min
        verbose=True
        # verbose=False
    )
    
    # Analyze results
    factorial.print_summary()
    factorial.plot_correlation_matrix()
    factorial.plot_main_effects('system_time_avg')
    factorial.plot_interaction_effects('system_time_avg', 'arrival_rate', 'num_troncos')
    
    # Export
    factorial.export_results()

    print("\n\nFactorial analysis examples completed!")
    print("Check the generated CSV files and plots for detailed results.")
    
    return factorial
    '''
# ================================================================

def pause_simulation(message="Continue? (Enter=yes / n=no): "):
    answer = input(message)
    if answer.lower().startswith('n'):
        print(f"Simulation stopped!")
        sys.exit()  # stops the simulation




def main():
    """Main example demonstrating refactored usage."""
    
    HOURS = 60  # Time conversion factor (base time: Minutos)
    DAYS = 1440
    YEARS = 525600
    
       
    # Create configuration
    config = SimulationConfig(
        warm_up_period=1*DAYS,
        duration=31*DAYS,   
        seed=123,#None,
        check_stability=True
    )
    config.validate()

    # Create event logger
    event_logger = EventLogger()
    
    # Build model
    print("Building model...")
    verbose = config.duration <= 1/10*HOURS    
    model = build_model(config.duration, event_logger, verbose=False)
    
    # Check stability BEFORE running (optional)
    print("\nChecking system stability...")
    stability_analyzer = StabilityAnalyzer(model)
    stability = stability_analyzer.check_system_stability()
    model.stability_result = stability
    
    # Run simulation
    print("\nRunning simulation (replication)...")
    model.run_simulation(
        validate_resources=True,  # Default True
        until=config.duration,
        seed=config.seed,
        warm_up_period=config.warm_up_period
    )        
    
    # === ANALYSIS PHASE (using separate modules) ===
    '''
    # ========================================
    # Trace specific chamada
    # ========================================    
    print("\n" + "="*80)
    print("FILTER: Journey of Chamada_1")
    print("="*80)    
    pause_simulation()
    model.trace_entity('Chamada_1')    
    
    
    # ========================================
    # Replay with filters
    # ========================================
    print("\n" + "="*80)
    print("FILTER: Replay - First 3 chamadas only")
    print("="*80)    
    pause_simulation()
    model.replay_trace(entity_pattern = r'^Chamada_[1-3]$')
    

    # ========================================
    # Trace specific resource
    # ========================================
    print("\n" + "="*80)
    print("FILTER: Replay - Troncos interactions only")
    print("="*80)    
    pause_simulation()
    model.replay_trace(resource_filter={'Troncos'})
    

    # ========================================
    # Trace specific event types
    # ========================================
    print("\n" + "="*80)
    print("FILTER: Replay - Queue and service events only")
    print("="*80)    
    pause_simulation()
    model.replay_trace(event_type_filter={'queue', 'service_start', 'service_end'})
    

    # ========================================
    # Trace time window
    # ========================================
    print("\n" + "="*80)
    print("FILTER: Replay - Events between t=2 and t=3")
    print("="*80)    
    pause_simulation()
    model.replay_trace(time_range=(2, 3))
    

    # ========================================
    # Combined filters
    # ========================================
    print("\n" + "="*80)
    print("FILTER: Replay - Chamada_1 at Troncos (queue + service)")
    print("="*80)    
    pause_simulation()
    model.replay_trace(
        entity_filter={'Chamada_1'},
        resource_filter={'Troncos'},
        event_type_filter={'queue', 'service_start', 'service_end'}
    )
    

    # ========================================
    # Multiple chamadas journeys
    # ========================================
    print("\n" + "="*80)
    print("FILTER: Detailed journeys of first 3 chamadas")
    print("="*80)    
    pause_simulation()
    model.trace_entities(['Chamada_1', 'Chamada_2', 'Chamada_3'])
    '''
    #'''
    # ========================================
    # Trace statistics
    # ========================================
    model.print_trace_statistics()
    pause_simulation()


    # ========================================
    # 2. Detailed reporting
    # ========================================
    reporter = SimulationReporter(model)
    reporter.print_results()
    
    # 3. Warm-up analysis
    print("\nAnalyzing warm-up period...")
    warmup_analyzer = WarmUpAnalyzer(model)
    warmup_analyzer.analyze_warm_up_period()
    
    # 4. Plotting
    print("\nPlotting resourse use over time...")
    plotter = SimulationPlotter(model)
    
    # Plot resource utilization over time
    plotter.plot_resource_use_over_time(show_warm_up=True, resource='escavadeira_FL', moving_average_window=50)
    plotter.plot_wip_over_time()
    plotter.plot_system_time_distribution()

    # Plot activity metrics
    print("\nPlotting activity metrics...")
    reporter._print_activity_metrics()
    plotter.plot_activity_metrics()

        
    # Plot resource utilization summary
    print("\nPlotting resourse summary...")
    plotter.plot_resources_utilization()
    reporter._print_resource_metrics()
    reporter._print_entity_counts()
    reporter._print_block_statistics()

    
    
    # Print variable results
    # ================================================================
    tracker = model.variable_tracker

    print(f"\n{'='*60}")
    print("RESULTADOS DO SBPO4")
    print(f"Seed usada: {config.seed}")
    print(f"{'='*60}")
    print(f"Total de entidades criadas: {model.entity_count}")
    print(f"Input minério: {tracker.get_final('input_minerio')}")
    print(f"Input estéril: {tracker.get_final('input_esteril')}")
    print(f"{'='*60}")


    def soma_material(material):
        return sum([
            tracker.get_final(f'proc_BS1_{material}')
        ])

    total_minerio = soma_material('minerio')
    total_esteril = soma_material('esteril')

    input_minerio = tracker.get_final('input_minerio')
    input_esteril = tracker.get_final('input_esteril')
    input_total = input_minerio + input_esteril

    proc_minerio = (
        tracker.get_final('proc_BS1_minerio')
    )

    proc_esteril = (
        tracker.get_final('proc_BS1_esteril')
    )

    proc_total = proc_minerio + proc_esteril

    print("\nVALIDAÇÃO DA PROPORÇÃO DO INPUT")
    print("=" * 60)

    if input_total > 0:
        print(f"Entrada - minério: {input_minerio} ({100*input_minerio/input_total:.2f}%)")
        print(f"Entrada - estéril: {input_esteril} ({100*input_esteril/input_total:.2f}%)")

    if proc_total > 0:
        print(f"Processado - minério: {proc_minerio} ({100*proc_minerio/proc_total:.2f}%)")
        print(f"Processado - estéril: {proc_esteril} ({100*proc_esteril/proc_total:.2f}%)")

    # ================================================================
    '''
    # Financial analysis
    print("\nPlotting financial analysys...")
    financial_analyzer = FinancialAnalyzer(model)
    financial_analyzer.print_financial_summary()
    financial_analyzer.plot_financial_breakdown()

    # 5. Export event log
    print("\nExporting event log...")
    df = event_logger.export_to_csv("results/ex3_event_log.csv")
    print(f"\nFirst 10 events:")
    print(df.head(10))
    
    # 6. Direct metrics access (if needed)
    metrics = MetricsCollector(model)
    entity_metrics = metrics.get_entity_metrics_summary()
    resource_metrics = metrics.get_resource_metrics_summary()
    
    print(f"\nAverage system time: {entity_metrics['tempo_medio_sistema']:.2f} min")
    print(f"Random seed for this run: {config.seed}")
    '''
    def div0(a, b):
        return a / b if b else 0

    carregamento_FL = model.blocks["Loading"]
    recupera_brita_transp_BS1 = model.blocks["Crushing"]

    sim_time = model.env.now

    def tamanho_medio_fila(block, tempo_final):
        data = getattr(block, "resource_data", [])

        if not data:
            return 0.0

        # Caso ProcessBlock / SyncProcessBlock:
        # [(time, in_service, queue_length), ...]
        if isinstance(data[0], tuple) and len(data[0]) >= 3:
            data = sorted(data, key=lambda x: x[0])

            area = 0.0

            for i in range(len(data) - 1):
                t_atual = data[i][0]
                fila_atual = data[i][2]
                t_prox = data[i + 1][0]

                if t_prox > t_atual:
                    area += fila_atual * (t_prox - t_atual)

            t_ultimo = data[-1][0]
            fila_ultima = data[-1][2]

            if tempo_final > t_ultimo:
                area += fila_ultima * (tempo_final - t_ultimo)

            return area / tempo_final if tempo_final > 0 else 0.0

        return 0.0

    falhas = tracker.get_final("kpi_falhas_caminhao")
    tempo_parado = tracker.get_final("kpi_tempo_parado_caminhao")

    mtbf_real = div0(
        tracker.get_final("kpi_soma_mtbf_real"),
        tracker.get_final("kpi_n_mtbf_real")
    )

    mttr_real = div0(
        tracker.get_final("kpi_soma_mttr_real"),
        tracker.get_final("kpi_n_mttr_real")
    )

    tempo_medio_carregando = div0(
        tracker.get_final("kpi_soma_tempo_carregando"),
        tracker.get_final("kpi_n_tempo_carregando")
    )

    tempo_medio_cheio = div0(
        tracker.get_final("kpi_soma_tempo_cheio"),
        tracker.get_final("kpi_n_tempo_cheio")
    )

    tempo_medio_descarregando = div0(
        tracker.get_final("kpi_soma_tempo_descarregando"),
        tracker.get_final("kpi_n_tempo_descarregando")
    )

    tempo_medio_fila_carregamento = div0(
        carregamento_FL.total_queue_time,
        carregamento_FL.entities_processed
    )

    tempo_medio_fila_descarregamento = div0(
        recupera_brita_transp_BS1.total_queue_time,
        recupera_brita_transp_BS1.entities_processed
    )

    # =========================
    # TEMPOS BÁSICOS
    # =========================
    tempo_total_frota = FROTA * sim_time

    tempo_parado = tracker.get_final("kpi_tempo_parado_caminhao")

    tempo_disponivel_frota = tempo_total_frota - tempo_parado

    tempo_ativo_frota = (
        tracker.get_final("kpi_soma_tempo_carregando")
        + tracker.get_final("kpi_soma_tempo_cheio")
        + tracker.get_final("kpi_soma_tempo_vazio")   # <-- NOVO
        + tracker.get_final("kpi_soma_tempo_descarregando")
    )

    # =========================
    # KPI's FROTA
    # =========================
    disponibilidade = 100 * div0(
        tempo_disponivel_frota,
        tempo_total_frota
    )

    utilizacao_calendario = 100 * div0(
        tempo_ativo_frota,
        tempo_total_frota
    )

    utilizacao_sobre_disponivel = 100 * div0(
        tempo_ativo_frota,
        tempo_disponivel_frota
    )

    tamanho_medio_fila_carregamento = tamanho_medio_fila(
        carregamento_FL,
        sim_time
    )

    taxa_carregamento = div0(
        carregamento_FL.entities_processed,
        sim_time
    )

    numero_medio_caminhoes_fila_carregamento = (
        taxa_carregamento * tempo_medio_fila_carregamento
    )

    tempo_medio_vazio = div0(
        tracker.get_final("kpi_soma_tempo_vazio"),
        tracker.get_final("kpi_n_tempo_vazio")
    )

    tempo_ocioso_disponivel = tempo_disponivel_frota - tempo_ativo_frota

    ociosidade_sobre_disponivel = 100 * div0(
        tempo_ocioso_disponivel,
        tempo_disponivel_frota
    )



    #tamanho_medio_fila_descarregamento = tamanho_medio_fila(
    #    recupera_brita_transp_BS1,
    #    sim_time
    #)
    tamanho_medio_fila_descarregamento = 0.0

    print("\n" + "=" * 60)
    print("KPI'S DA FROTA")
    print(f"{FROTA}_{'AUTO' if AUTONOMO else 'CONV'}_")
    print("=" * 60)
    print(f"Número total de falhas: {falhas}")
    print(f"Tempo total de simulação: {sim_time:.2f} min ({sim_time/60:.2f} h | {sim_time/1440:.2f} dias)")
    print(f"Disponibilidade física média da frota: {disponibilidade:.2f}%")
    print(f"Utilização sobre disponivel média da frota: {utilizacao_sobre_disponivel:.2f}%")
    print(f"Utilização calendario média da frota: {utilizacao_calendario:.2f}%")
    print(f"Tempo ocioso/espera disponível da frota: {tempo_ocioso_disponivel:.2f} min")
    print(f"Ociosidade/espera sobre disponível: {ociosidade_sobre_disponivel:.2f}%")
    print(f"Número de amostras usadas no MTBF real: {tracker.get_final('kpi_n_mtbf_real')}")
    print(f"MTBF médio real: {mtbf_real:.2f} min")
    print(f"MTTR médio real: {mttr_real:.2f} min")
    print(f"Toneladas de minério processadas: {tracker.get_final('kpi_ton_minerio'):.0f} t")
    print(f"Toneladas de estéril processadas: {tracker.get_final('kpi_ton_esteril'):.0f} t")
    print(f"Tempo médio carregando: {tempo_medio_carregando:.2f} min")
    print(f"Tempo médio deslocando cheio: {tempo_medio_cheio:.2f} min")
    print(f"Tempo médio deslocando vazio: {tempo_medio_vazio:.2f} min")
    print(f"Tempo médio descarregando: {tempo_medio_descarregando:.2f} min")
    print(f"Tempo médio dos caminhões em fila no carregamento: {tempo_medio_fila_carregamento:.2f} min")
    print(f"Tempo médio do ROM aguardando em estoque pra ser processado: {tempo_medio_fila_descarregamento:.2f} min")
    print(f"Número médio de caminhões em fila no carregamento: {numero_medio_caminhoes_fila_carregamento:.2f} caminhões")
    print("=" * 60)
    print(f"Entidades carregadas: {carregamento_FL.entities_processed}")
    print(f"Entidades britadas/processadas: {recupera_brita_transp_BS1.entities_processed}")
    print(f"Tempo médio aguardando britagem: {tempo_medio_fila_descarregamento:.2f} min")
    #print(f"Tamanho médio da fila de caminhões no carregamento: {tamanho_medio_fila_carregamento:.2f} entidades")
    #print(f"Tamanho médio da fila no descarregamento: {tamanho_medio_fila_descarregamento:.2f} entidades")
    
    return model, event_logger
    

# ===========================================
# Simulation Kit
# ===========================================
def run_single_replication():
    return main()


def run_replications_cli():
    run_replications()


#def run_factorial_cli():
#    return factorial_analysis()


def run_visualization_cli(simulation_time=600):
    return run_visualization(build_model, simulation_time=simulation_time)
# ===========================================