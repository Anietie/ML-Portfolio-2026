#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <numeric>
#include <optional>
#include <string.h>
#include <cstdlib>

namespace py = pybind11;

class GameEnvCpp
{
public:
    // Game environment implementation in C++ for performance
    std::vector<std::vector<int>> board;
    std::vector<int> captured_seeds;
    int current_player;
    int current_step;
    int episode_count;
    int max_steps;

    GameEnvCpp()
    {
        board = std::vector<std::vector<int>>(2, std::vector<int>(6, 4));
        captured_seeds = std::vector<int>(2, 0);
        current_player = 0;
        current_step = 0;
        episode_count = 0;
        max_steps = 200;
    }

    py::array_t<int> get_board_array() const
    {
        auto result = py::array_t<int>({2, 6});
        auto ptr = result.mutable_unchecked<2>();
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                ptr(i, j) = board[i][j];
            }
        }
        return result;
    }

    bool _would_starve_opponent(int player, int action) const
    {
        if (board[player][action] == 0)
            return true;

        auto temp_board = board;
        int seeds_to_sow = temp_board[player][action];
        int seed_count = seeds_to_sow;
        temp_board[player][action] = 0;

        int r = player, c = action;
        int orig_r = r, orig_c = c;
        int opponent_row = 1 - player;
        bool first_loop = true;

        while (seeds_to_sow > 0)
        {
            if (r == 0 && c > 0)
                c--;
            else if (r == 0 && c == 0)
                r = 1;
            else if (r == 1 && c < 5)
                c++;
            else if (r == 1 && c == 5)
            {
                r = 0;
                c = 5;
            }

            if (seed_count >= 12 && first_loop && orig_r == r && orig_c == c)
            {
                first_loop = false;
                continue;
            }
            temp_board[r][c]++;
            seeds_to_sow--;
        }

        int opp_sum = 0;
        for (int i = 0; i < 6; i++)
        {
            opp_sum += temp_board[opponent_row][i];
        }
        return opp_sum == 0;
    }

    bool _has_non_starving_move(int player) const
    {
        for (int a = 0; a < 6; a++)
        {
            if (board[player][a] > 0 && !_would_starve_opponent(player, a))
                return true;
        }
        return false;
    }

    std::vector<int> valid_moves() const
    {
        bool has_non_starving = _has_non_starving_move(current_player);
        std::vector<int> legal;
        for (int a = 0; a < 6; a++)
        {
            if (board[current_player][a] == 0)
                continue;
            if (has_non_starving && _would_starve_opponent(current_player, a))
                continue;
            legal.push_back(a);
        }
        return legal;
    }

    bool _check_game_over() const
    {
        int p_sum = 0;
        for (int i = 0; i < 6; i++)
            p_sum += board[current_player][i];
        if (p_sum == 0)
            return true;
        if (captured_seeds[0] >= 25 || captured_seeds[1] >= 25)
            return true;
        return false;
    }

    int _simulate_capture(int player, int r, int c, const std::vector<std::vector<int>> &b) const
    {
        auto temp_board = b;
        int seeds_cap = 0;

        if (temp_board[r][c] == 2 || temp_board[r][c] == 3)
        {
            seeds_cap += temp_board[r][c];
            temp_board[r][c] = 0;
            int pit_number = c;

            if (player == 1)
            {
                while (pit_number < 5 && (temp_board[0][pit_number + 1] == 2 || temp_board[0][pit_number + 1] == 3))
                {
                    seeds_cap += temp_board[0][pit_number + 1];
                    temp_board[0][pit_number + 1] = 0;
                    pit_number++;
                }
            }
            else
            {
                while (pit_number > 0 && (temp_board[1][pit_number - 1] == 2 || temp_board[1][pit_number - 1] == 3))
                {
                    seeds_cap += temp_board[1][pit_number - 1];
                    temp_board[1][pit_number - 1] = 0;
                    pit_number--;
                }
            }
        }
        return seeds_cap;
    }

    int _capture_seeds(int player, int r, int c)
    {
        int captured = 0;

        if (board[r][c] == 2 || board[r][c] == 3)
        {
            captured += board[r][c];
            board[r][c] = 0;
            int pit_number = c;

            if (player == 1)
            {
                while (pit_number < 5 && (board[0][pit_number + 1] == 2 || board[0][pit_number + 1] == 3))
                {
                    captured += board[0][pit_number + 1];
                    board[0][pit_number + 1] = 0;
                    pit_number++;
                }
            }
            else
            {
                while (pit_number > 0 && (board[1][pit_number - 1] == 2 || board[1][pit_number - 1] == 3))
                {
                    captured += board[1][pit_number - 1];
                    board[1][pit_number - 1] = 0;
                    pit_number--;
                }
            }
            captured_seeds[player] += captured;
        }
        return captured;
    }

    py::dict _build_info(int acting_player, int seeds_captured, bool is_terminal) const
    {
        py::dict info;
        int winner = -2;
        if (is_terminal)
        {
            if (captured_seeds[0] > captured_seeds[1])
                winner = 0;
            else if (captured_seeds[1] > captured_seeds[0])
                winner = 1;
            else
                winner = -1;
        }

        info["current_player"] = current_player;
        info["acting_player"] = acting_player;
        info["score_0"] = captured_seeds[0];
        info["score_1"] = captured_seeds[1];
        info["ply"] = current_step;
        info["seeds_captured"] = seeds_captured;
        info["is_terminal"] = is_terminal;
        info["winner"] = winner;
        return info;
    }

    py::tuple reset(std::optional<int> seed = std::nullopt)
    {
        if (seed.has_value())
            srand(seed.value());

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 6; j++)
                board[i][j] = 4;
        }
        captured_seeds[0] = 0;
        captured_seeds[1] = 0;

        current_player = episode_count % 2;
        episode_count++;
        current_step = 0;

        return py::make_tuple(get_board_array(), _build_info(current_player, 0, false));
    }

    py::tuple step(int action)
    {
        int player = current_player;
        int seeds_captured_this_step = 0;
        double reward = 0.0;

        if (board[player][action] == 0 || (_has_non_starving_move(player) && _would_starve_opponent(player, action)))
        {
            current_player = 1 - player;
            current_step++;
            bool done = _check_game_over();
            bool truncated = current_step >= max_steps;
            return py::make_tuple(get_board_array(), -1.0, done, truncated, _build_info(player, 0, done || truncated));
        }

        int seeds_to_sow = board[player][action];
        int seed_count = seeds_to_sow;
        board[player][action] = 0;

        int r = player, c = action, orig_r = r, orig_c = c;
        bool first_loop = true;

        while (seeds_to_sow > 0)
        {
            if (r == 0 && c > 0)
                c--;
            else if (r == 0 && c == 0)
                r = 1;
            else if (r == 1 && c < 5)
                c++;
            else if (r == 1 && c == 5)
            {
                r = 0;
                c = 5;
            }

            if (seed_count >= 12 && first_loop && orig_r == r && orig_c == c)
            {
                first_loop = false;
                continue;
            }
            board[r][c]++;
            seeds_to_sow--;
        }

        int opponent_row = 1 - player;
        if ((player == 1 && r == 0 && (board[r][c] == 2 || board[r][c] == 3)) ||
            (player == 0 && r == 1 && (board[r][c] == 2 || board[r][c] == 3)))
        {

            int sim_capture = _simulate_capture(player, r, c, board);
            int opp_sum = 0;
            for (int i = 0; i < 6; i++)
                opp_sum += board[opponent_row][i];

            if (opp_sum - sim_capture > 0)
            {
                seeds_captured_this_step = _capture_seeds(player, r, c);
            }
        }

        current_player = 1 - player;
        current_step++;

        bool done = _check_game_over();
        bool truncated = current_step >= max_steps;

        if (done || truncated)
        {
            reward = (double(captured_seeds[player]) - double(captured_seeds[1 - player])) / 48.0;
        }

        return py::make_tuple(get_board_array(), reward, done, truncated, _build_info(player, seeds_captured_this_step, done || truncated));
    }

    GameEnvCpp clone() const
    {
        GameEnvCpp copy_env;
        copy_env.board = board;
        copy_env.captured_seeds = captured_seeds;
        copy_env.current_player = current_player;
        copy_env.current_step = current_step;
        copy_env.episode_count = episode_count;
        return copy_env;
    }

    py::dict get_state() const
    {
        py::dict state;
        state["board"] = get_board_array();

        auto cap_array = py::array_t<int>(2);
        auto ptr = cap_array.mutable_unchecked<1>();
        ptr(0) = captured_seeds[0];
        ptr(1) = captured_seeds[1];
        state["captured_seeds"] = cap_array;

        state["current_player"] = current_player;
        state["current_step"] = current_step;
        return state;
    }

    void set_state(py::dict state)
    {
        auto b_array = state["board"].cast<py::array_t<int>>();
        auto b_ptr = b_array.unchecked<2>();
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 6; j++)
                board[i][j] = b_ptr(i, j);
        }

        auto cap_array = state["captured_seeds"].cast<py::array_t<int>>();
        auto cap_ptr = cap_array.unchecked<1>();
        captured_seeds[0] = cap_ptr(0);
        captured_seeds[1] = cap_ptr(1);

        current_player = state["current_player"].cast<int>();
        current_step = state["current_step"].cast<int>();
    }
};

PYBIND11_MODULE(cpp_env, m)
{
    py::class_<GameEnvCpp>(m, "GameEnvCpp")
        .def(py::init<>())
        .def("reset", &GameEnvCpp::reset, py::arg("seed") = py::none())
        .def("step", &GameEnvCpp::step, py::arg("action"))
        .def("valid_moves", &GameEnvCpp::valid_moves)
        .def("clone", &GameEnvCpp::clone)
        .def("get_state", &GameEnvCpp::get_state)
        .def("set_state", &GameEnvCpp::set_state)
        .def_readwrite("current_player", &GameEnvCpp::current_player)
        .def_readwrite("current_step", &GameEnvCpp::current_step)
        .def_readwrite("episode_count", &GameEnvCpp::episode_count)
        .def_property_readonly("board", &GameEnvCpp::get_board_array)
        .def_property_readonly("captured_seeds", [](const GameEnvCpp &env)
                               {
            auto cap_array = py::array_t<int>(2);
            auto ptr = cap_array.mutable_unchecked<1>();
            ptr(0) = env.captured_seeds[0];
            ptr(1) = env.captured_seeds[1];
            return cap_array; });
}