// engine.cpp
// Data generator for Nsa Isong (Ayo) game — uses minimax search to build training data
// Board is 2 rows x 6 cols, counter-clockwise sowing
// Outputs CSV with board state, move policy, and value scores
//
// Build: g++ -O3 -std=c++17 -march=native -o engine engine.cpp
// Run: ./engine <num_positions> <depth> <output.csv>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

// game constants
static constexpr int ROWS = 2;
static constexpr int COLS = 6;
static constexpr int TOTAL_SEEDS = 48;
static constexpr int MAX_PLIES = 200;
static constexpr int NO_CAPTURE_MAX = 80;
static constexpr int MAX_SEEDS_HOLE = 52;
static constexpr int INF = 1'000'000;

// zobrist hashing for transposition table
static uint64_t ZOB[ROWS][COLS][MAX_SEEDS_HOLE];
static uint64_t ZOB_TURN;

void initZobrist()
{
    std::mt19937_64 rng(0xA0B0A2D2026AULL);
    for (auto &rr : ZOB)
        for (auto &cc : rr)
            for (auto &v : cc)
                v = rng();
    ZOB_TURN = rng();
}

// board representation
struct Board
{
    int b[ROWS][COLS];  // seed counts
    int score[2];       // captured seeds per player
    int turn;           // 0 or 1
    int ply;            // total moves in this game
    int no_capture_ply; // consecutive plies without a capture
    uint64_t zobrist;   // incremental hash
};

// compute zobrist hash for a position
uint64_t computeZobrist(const Board &bd)
{
    uint64_t h = 0;
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++)
            h ^= ZOB[r][c][bd.b[r][c]];
    if (bd.turn == 1)
        h ^= ZOB_TURN;
    return h;
}

Board makeInitial(int starting_player)
{
    Board bd{};
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++)
            bd.b[r][c] = 4;
    bd.score[0] = 0;
    bd.score[1] = 0;
    bd.turn = starting_player;
    bd.ply = 0;
    bd.no_capture_ply = 0;
    bd.zobrist = computeZobrist(bd);
    return bd;
}

// next hole in counter-clockwise sowing
inline std::pair<int, int> nextPos(int row, int col)
{
    if (row == 0)
        return (col > 0) ? std::make_pair(0, col - 1)
                         : std::make_pair(1, 0);
    else
        return (col < 5) ? std::make_pair(1, col + 1)
                         : std::make_pair(0, 5);
}

// simulate a sow without modifying board state
struct SowResult
{
    int tb[ROWS][COLS];
    int land_row, land_col;
};

SowResult simulateSow(const Board &bd, int player, int action)
{
    SowResult sr{};
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++)
            sr.tb[r][c] = bd.b[r][c];

    int seeds = sr.tb[player][action];
    int seed_count = seeds;
    sr.tb[player][action] = 0;

    int row = player, col = action;
    bool first_loop = true;

    while (seeds > 0)
    {
        auto [nr, nc] = nextPos(row, col);
        row = nr;
        col = nc;

        if (seed_count >= 12 && first_loop &&
            row == player && col == action)
        {
            first_loop = false;
            continue;
        }

        sr.tb[row][col]++;
        seeds--;
    }

    sr.land_row = row;
    sr.land_col = col;
    return sr;
}

// starvation rules - can't leave opponent with no moves
bool wouldStarveOpponent(const Board &bd, int player, int action)
{
    if (bd.b[player][action] == 0)
        return true;
    SowResult sr = simulateSow(bd, player, action);
    int opp = 1 - player, total = 0;
    for (int c = 0; c < COLS; c++)
        total += sr.tb[opp][c];
    return total == 0;
}

bool hasNonStarvingMove(const Board &bd, int player)
{
    for (int a = 0; a < COLS; a++)
        if (bd.b[player][a] > 0 && !wouldStarveOpponent(bd, player, a))
            return true;
    return false;
}

// get all legal moves for current player
std::vector<int> validMoves(const Board &bd)
{
    int p = bd.turn;
    bool hasNS = hasNonStarvingMove(bd, p);
    std::vector<int> moves;
    moves.reserve(COLS);
    for (int a = 0; a < COLS; a++)
    {
        if (bd.b[p][a] == 0)
            continue;
        if (hasNS && wouldStarveOpponent(bd, p, a))
            continue;
        moves.push_back(a);
    }
    return moves;
}

// check what would happen if we captured (no board modification)
struct CaptureResult
{
    int seeds_captured;
    bool would_empty_opponent;
};

CaptureResult simulateCapture(const int tb[ROWS][COLS],
                              int player, int land_row, int land_col)
{
    int sim[ROWS][COLS];
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++)
            sim[r][c] = tb[r][c];

    int captured = 0;
    if (sim[land_row][land_col] == 2 || sim[land_row][land_col] == 3)
    {
        captured += sim[land_row][land_col];
        sim[land_row][land_col] = 0;
        int pit = land_col;

        if (player == 1)
        {
            while (pit < 5 &&
                   (sim[0][pit + 1] == 2 || sim[0][pit + 1] == 3))
            {
                captured += sim[0][pit + 1];
                sim[0][pit + 1] = 0;
                pit++;
            }
        }
        else
        {
            while (pit > 0 &&
                   (sim[1][pit - 1] == 2 || sim[1][pit - 1] == 3))
            {
                captured += sim[1][pit - 1];
                sim[1][pit - 1] = 0;
                pit--;
            }
        }
    }

    int opp = 1 - player, opp_total = 0;
    for (int c = 0; c < COLS; c++)
        opp_total += sim[opp][c];
    return {captured, opp_total == 0};
}

// apply a move and return new board (original is unchanged)
Board applyMove(const Board &bd, int action)
{
    Board nb = bd;
    int player = bd.turn;

    // pick up seeds and sow them
    int seeds = nb.b[player][action];
    int seed_count = seeds;

    nb.zobrist ^= ZOB[player][action][nb.b[player][action]];
    nb.b[player][action] = 0;
    nb.zobrist ^= ZOB[player][action][0];

    int row = player, col = action;
    bool first_loop = true;
    int land_row = player, land_col = action;

    while (seeds > 0)
    {
        auto [nr, nc] = nextPos(row, col);
        row = nr;
        col = nc;

        // skip starting hole on second pass if we had lots of seeds
        if (seed_count >= 12 && first_loop && row == player && col == action)
        {
            first_loop = false;
            continue;
        }

        nb.zobrist ^= ZOB[row][col][nb.b[row][col]];
        nb.b[row][col]++;
        nb.zobrist ^= ZOB[row][col][nb.b[row][col]];
        seeds--;
        land_row = row;
        land_col = col;
    }

    // now handle captures if applicable
    bool trigger =
        (player == 1 && land_row == 0 && (nb.b[0][land_col] == 2 || nb.b[0][land_col] == 3)) ||
        (player == 0 && land_row == 1 && (nb.b[1][land_col] == 2 || nb.b[1][land_col] == 3));

    bool captured_this_move = false;
    if (trigger)
    {
        CaptureResult cr = simulateCapture(reinterpret_cast<const int (*)[COLS]>(nb.b),
                                           player, land_row, land_col);
        if (!cr.would_empty_opponent && cr.seeds_captured > 0)
        {
            int pit = land_col;
            nb.zobrist ^= ZOB[land_row][land_col][nb.b[land_row][land_col]];
            nb.score[player] += nb.b[land_row][land_col];
            nb.b[land_row][land_col] = 0;
            nb.zobrist ^= ZOB[land_row][land_col][0];

            if (player == 1)
            {
                while (pit < 5 &&
                       (nb.b[0][pit + 1] == 2 || nb.b[0][pit + 1] == 3))
                {
                    nb.zobrist ^= ZOB[0][pit + 1][nb.b[0][pit + 1]];
                    nb.score[player] += nb.b[0][pit + 1];
                    nb.b[0][pit + 1] = 0;
                    nb.zobrist ^= ZOB[0][pit + 1][0];
                    pit++;
                }
            }
            else
            {
                while (pit > 0 && (nb.b[1][pit - 1] == 2 || nb.b[1][pit - 1] == 3))
                {
                    nb.zobrist ^= ZOB[1][pit - 1][nb.b[1][pit - 1]];
                    nb.score[player] += nb.b[1][pit - 1];
                    nb.b[1][pit - 1] = 0;
                    nb.zobrist ^= ZOB[1][pit - 1][0];
                    pit--;
                }
            }
            captured_this_move = true;
        }
    }

    nb.ply++;
    nb.no_capture_ply = captured_this_move ? 0 : bd.no_capture_ply + 1;
    nb.turn = 1 - player;
    nb.zobrist ^= ZOB_TURN;
    return nb;
}

// is the game finished?
bool isTerminal(const Board &bd)
{
    if (bd.score[0] + bd.score[1] == TOTAL_SEEDS)
        return true;
    if (bd.ply >= MAX_PLIES)
        return true;
    if (bd.no_capture_ply >= NO_CAPTURE_MAX)
        return true;
    if (validMoves(bd).empty())
        return true;
    return false;
}

// evaluate board (from player 0's perspective)
int evaluate(const Board &bd)
{
    int score_diff = bd.score[0] - bd.score[1];

    int s0 = 0, s1 = 0;
    for (int c = 0; c < COLS; c++)
    {
        s0 += bd.b[0][c];
        s1 += bd.b[1][c];
    }

    // Capturable holes: opponent holes with exactly 2 or 3 seeds
    int cap0 = 0, cap1 = 0;
    for (int c = 0; c < COLS; c++)
    {
        if (bd.b[1][c] == 2 || bd.b[1][c] == 3)
            cap0++;
        if (bd.b[0][c] == 2 || bd.b[0][c] == 3)
            cap1++;
    }

    int mob0 = 0, mob1 = 0;
    for (int c = 0; c < COLS; c++)
    {
        if (bd.b[0][c] > 0)
            mob0++;
        if (bd.b[1][c] > 0)
            mob1++;
    }

    int progress = bd.score[0] + bd.score[1];

    return score_diff * 100 + (s0 - s1) * 3 + (cap0 - cap1) * 15 + (mob0 - mob1) * 5 + progress * 1;
}

// transposition table; caches search results
enum class TTFlag : uint8_t
{
    EXACT,
    LOWER,
    UPPER
};

struct TTEntry
{
    uint64_t key = 0;
    int value = 0;
    int depth = 0;
    TTFlag flag = TTFlag::EXACT;
    int best_move = -1;
};

// 4M entries (~128 MB). Raise to 1<<23 if you have 16 GB+ RAM.
static constexpr size_t TT_SIZE = (1u << 22);
static TTEntry TT[TT_SIZE];

inline TTEntry *ttLookup(uint64_t key)
{
    return &TT[key & (TT_SIZE - 1)];
}

void ttStore(uint64_t key, int value, int depth, TTFlag flag, int bm)
{
    TTEntry *e = ttLookup(key);
    e->key = key;
    e->value = value;
    e->depth = depth;
    e->flag = flag;
    e->best_move = bm;
}

// negamax with alpha-beta pruning and TT
int negamax(const Board &bd, int depth, int alpha, int beta, int color)
{
    if (isTerminal(bd))
    {
        float fval = static_cast<float>(bd.score[0] - bd.score[1]) / 48.0f;
        return static_cast<int>(color * fval * INF);
    }
    if (depth == 0)
        return color * evaluate(bd);

    TTEntry *tte = ttLookup(bd.zobrist);
    if (tte->key == bd.zobrist && tte->depth >= depth)
    {
        switch (tte->flag)
        {
        case TTFlag::EXACT:
            return tte->value;
        case TTFlag::LOWER:
            alpha = std::max(alpha, tte->value);
            break;
        case TTFlag::UPPER:
            beta = std::min(beta, tte->value);
            break;
        }
        if (alpha >= beta)
            return tte->value;
    }

    auto moves = validMoves(bd);

    // Move ordering: TT best move first
    if (tte->key == bd.zobrist && tte->best_move >= 0)
    {
        auto it = std::find(moves.begin(), moves.end(), tte->best_move);
        if (it != moves.end())
            std::rotate(moves.begin(), it, it + 1);
    }

    int orig_alpha = alpha;
    int best_val = -INF;
    int best_move = moves.empty() ? -1 : moves[0];

    for (int mv : moves)
    {
        Board child = applyMove(bd, mv);
        int val = -negamax(child, depth - 1, -beta, -alpha, -color);
        if (val > best_val)
        {
            best_val = val;
            best_move = mv;
        }
        alpha = std::max(alpha, val);
        if (alpha >= beta)
            break;
    }

    TTFlag flag;
    if (best_val <= orig_alpha)
        flag = TTFlag::UPPER;
    else if (best_val >= beta)
        flag = TTFlag::LOWER;
    else
        flag = TTFlag::EXACT;
    ttStore(bd.zobrist, best_val, depth, flag, best_move);

    return best_val;
}

// MTD(f) search - iterative deepening with zero-window searches
int mtdf(const Board &bd, int f, int depth)
{
    int g = f, upper = INF, lower = -INF;
    while (lower < upper)
    {
        int beta = (g == lower) ? g + 1 : g;
        g = negamax(bd, depth, beta - 1, beta,
                    bd.turn == 0 ? 1 : -1);
        if (g < beta)
            upper = g;
        else
            lower = g;
    }
    return g;
}

int idMTDF(const Board &bd, int max_depth)
{
    int f = 0;
    for (int d = 1; d <= max_depth; d++)
        f = mtdf(bd, f, d);
    return f;
}

// compute everything we need to train all 5 model heads
struct PositionTargets
{
    std::array<float, COLS> policy; // softmax over 6 move slots
    float value;                    // MTD(f)/INF, acting player POV
    int seed[ROWS][COLS];           // exact hole counts
    int capture_next;               // 1 if best move captures
    int next_b[ROWS][COLS];         // board after best move
};

PositionTargets computeTargets(const Board &bd, int depth)
{
    PositionTargets t{};

    // copy raw hole counts for seed counter training
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++)
            t.seed[r][c] = bd.b[r][c];

    // evaluate each legal move
    auto moves = validMoves(bd);
    std::array<float, COLS> raw{};
    raw.fill(-1e9f);

    int color = (bd.turn == 0) ? 1 : -1;
    for (int mv : moves)
    {
        Board child = applyMove(bd, mv);
        int val = -negamax(child, depth - 1, -INF, INF, -color);
        raw[mv] = static_cast<float>(val);
    }

    // run MTD(f) for exact evaluation
    int raw_val = idMTDF(bd, depth);
    float signed_val = static_cast<float>(raw_val) / static_cast<float>(INF);
    t.value = std::max(-1.f, std::min(1.f, signed_val));

    // find best move
    int best_move = moves[0];
    float best_raw = raw[moves[0]];
    for (int mv : moves)
        if (raw[mv] > best_raw)
        {
            best_raw = raw[mv];
            best_move = mv;
        }

    // compute policy with softmax
    float sum = 0.f;
    for (int i = 0; i < COLS; i++)
    {
        if (raw[i] > -1e8f)
        {
            t.policy[i] = std::exp((raw[i] - best_raw) / 100.f);
            sum += t.policy[i];
        }
    }
    if (sum > 0.f)
        for (auto &p : t.policy)
            p /= sum;

    // apply best move to get next board
    Board next = applyMove(bd, best_move);
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++)
            t.next_b[r][c] = next.b[r][c];

    // check if best move captured anything
    int before = 0, after = 0;
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++)
        {
            before += bd.b[r][c];
            after += next.b[r][c];
        }
    t.capture_next = (after < before) ? 1 : 0;

    return t;
}

// write CSV row with all position data
void writeRow(std::ofstream &csv, const Board &bd, const PositionTargets &t)
{
    // board state
    for (int c = 0; c < COLS; c++)
        csv << bd.b[0][c] << ",";
    for (int c = 0; c < COLS; c++)
        csv << bd.b[1][c] << ",";

    // policy
    for (int i = 0; i < COLS; i++)
        csv << t.policy[i] << ",";

    // value score
    csv << t.value << ",";

    // seed counts (for seed counter head training)
    for (int c = 0; c < COLS; c++)
        csv << t.seed[0][c] << ",";
    for (int c = 0; c < COLS; c++)
        csv << t.seed[1][c] << ",";

    // capture flag
    csv << t.capture_next << ",";

    // next board state
    for (int c = 0; c < COLS; c++)
        csv << t.next_b[0][c] << ",";
    for (int c = 0; c < COLS - 1; c++)
        csv << t.next_b[1][c] << ",";
    csv << t.next_b[1][COLS - 1] << ",";

    // metadata
    csv << bd.turn << ","
        << bd.score[0] << ","
        << bd.score[1] << ","
        << bd.ply << "\n";
}

// generate training dataset
void generateDataset(int num_positions, int depth, const std::string &out_path)
{
    std::ofstream csv(out_path);
    if (!csv.is_open())
    {
        std::cerr << "Cannot open: " << out_path << "\n";
        return;
    }

    // CSV Header
    // Board state
    for (int i = 0; i < 12; i++)
        csv << "hole_" << i << ",";
    // Policy Head
    for (int i = 0; i < COLS; i++)
        csv << "policy_" << i << ",";
    // Value Head
    csv << "value,";
    // Seed Counter Head — explicitly named to distinguish from hole_*
    for (int i = 0; i < 12; i++)
        csv << "seed_" << i << ",";
    // Capture Predictor Head
    csv << "capture_next,";
    // Next-State Head
    for (int i = 0; i < 12; i++)
        csv << "next_hole_" << i << ",";
    // Context
    csv << "turn,score_0,score_1,ply\n";

    // RNG for starting player and move diversification
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> coin(0, 1);

    int written = 0, games = 0;
    auto t0 = std::chrono::steady_clock::now();

    while (written < num_positions)
    {
        // Randomise starting player each game.
        // This prevents first-mover bias in the dataset and ensures
        // both perspectives are represented equally over time.
        int starting_player = coin(rng);
        Board bd = makeInitial(starting_player);
        games++;

        for (;;)
        {
            if (isTerminal(bd))
                break;

            auto moves = validMoves(bd);
            if (moves.empty())
                break;

            // Compute all targets for this position
            PositionTargets tgt = computeTargets(bd, depth);
            writeRow(csv, bd, tgt);
            written++;

            if (written >= num_positions)
                break;

            // Random move to diversify positions across the dataset.
            // Playing the best move every time would over-sample
            // narrow lines the engine happens to prefer.
            std::uniform_int_distribution<int> dist(
                0, (int)moves.size() - 1);
            bd = applyMove(bd, moves[dist(rng)]);
        }

        // Progress every 10k
        if (written % 10'000 == 0 && written > 0)
        {
            auto elapsed = std::chrono::duration_cast<
                               std::chrono::seconds>(
                               std::chrono::steady_clock::now() - t0)
                               .count();
            double rate = elapsed > 0
                              ? static_cast<double>(written) / elapsed
                              : 0.0;
            double eta = rate > 0
                             ? (num_positions - written) / rate / 3600.0
                             : 0.0;
            std::cerr
                << "[" << written << "/" << num_positions << "]"
                << "  games=" << games
                << "  rate=" << static_cast<int>(rate) << "/s"
                << "  elapsed=" << elapsed << "s"
                << "  ETA≈" << eta << "h\n";
        }
    }

    csv.close();
    std::cerr << "\nDone.\n"
              << "  positions : " << written << "\n"
              << "  games     : " << games << "\n"
              << "  output    : " << out_path << "\n";
}

// entry point
int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cerr
            << "Ayo Engine — God Teacher Data Generator\n\n"
            << "Usage  : ./ayo_engine <num_positions> <depth> <output.csv>\n\n"
            << "Examples:\n"
            << "  ./ayo_engine 50000   6  test.csv\n"
            << "  ./ayo_engine 500000  8  dev.csv\n"
            << "  ./ayo_engine 5000000 8  dataset.csv\n\n"
            << "Depth guide (Core i5 5th gen):\n"
            << "  depth 6  ~0.5ms/pos  — fast, good for testing\n"
            << "  depth 8  ~2-3ms/pos  — recommended for production\n"
            << "  depth 10 ~8-12ms/pos — high quality, very slow\n\n"
            << "Output columns (28 total):\n"
            << "  hole_0..11       board state\n"
            << "  policy_0..5      Policy Head target\n"
            << "  value            Value Head target\n"
            << "  seed_0..11       Seed Counter Head target\n"
            << "  capture_next     Capture Predictor Head target\n"
            << "  next_hole_0..11  Next-State Head target\n"
            << "  turn, score_0, score_1, ply  context\n";
        return 1;
    }

    int num_pos = std::stoi(argv[1]);
    int depth = std::stoi(argv[2]);
    std::string out_path = argv[3];

    std::cerr << "Ayo Engine — God Teacher Data Generator\n"
              << "  positions   : " << num_pos << "\n"
              << "  depth       : " << depth << "\n"
              << "  output      : " << out_path << "\n"
              << "  ply cap     : " << MAX_PLIES << "\n"
              << "  cycle limit : " << NO_CAPTURE_MAX << " plies\n"
              << "  TT entries  : " << TT_SIZE / 1'000'000 << "M\n"
              << "  start player: randomised per game\n\n";

    initZobrist();
    std::memset(TT, 0, sizeof(TT));

    generateDataset(num_pos, depth, out_path);
    return 0;
}