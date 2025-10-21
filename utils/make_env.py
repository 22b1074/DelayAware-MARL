def make_env(scenario_name, discrete_action=False):
    from pettingzoo.mpe.simple_speaker_listener_v4 import parallel_env as ssl_env
    from pettingzoo.mpe.simple_spread_v3 import parallel_env as ss_env
    from pettingzoo.mpe.simple_reference_v3 import parallel_env as sr_env

    scenario_dict = {
        'simple_speaker_listener': ssl_env,
        'simple_spread': ss_env,
        'simple_reference': sr_env,
    }

    if scenario_name not in scenario_dict:
        raise ValueError(f"Scenario {scenario_name} not found in MPE2 environments")

    env = scenario_dict[scenario_name](
        max_cycles=25,
        continuous_actions=not discrete_action
    )

    return env
