package mahe.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.text.BasicText
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import androidx.compose.ui.window.rememberWindowState

fun main() {
    Runtime.getRuntime().addShutdownHook(Thread { EngineLauncher.stop() })
    application {
        val client = remember { UiClient() }
        LaunchedEffect(Unit) {
            EngineLauncher.ensureRunning()
            client.connect(this)
        }

        Window(
            onCloseRequest = {
                EngineLauncher.stop()
                exitApplication()
            },
            title = "MAHE // 分布式训练控制终端",
            state = rememberWindowState(width = 1100.dp, height = 720.dp),
        ) {
            var booted by remember { mutableStateOf(false) }
            androidx.compose.material3.MaterialTheme(colorScheme = MaheTheme.colors) {
                Box(Modifier.fillMaxSize().background(MaheTheme.bg)) {
                    if (!booted) {
                        BootSequence(onDone = { booted = true })
                    } else {
                        MainScreen(client)
                    }
                }
            }
        }
    }
}

@Composable
fun MainScreen(client: UiClient) {
    val events by client.events.collectAsState()
    val connected by client.connected.collectAsState()
    val rounds = events.filter { it.type == "round" }
    val nodes = events.filter { it.type == "node_join" }

    Column(Modifier.fillMaxSize().padding(16.dp)) {
        // 顶栏
        Row(verticalAlignment = Alignment.CenterVertically) {
            BasicText("MAHE CONTROL TERMINAL", style = MaheTheme.title)
            Spacer(Modifier.weight(1f))
            BasicText(
                if (connected) "◉ LINKED" else "◌ OFFLINE",
                style = MaheTheme.mono.copy(
                    color = if (connected) MaheTheme.ok else MaheTheme.dead
                ),
            )
        }
        Spacer(Modifier.height(12.dp))

        Row(Modifier.weight(1f)) {
            // 左：指标 + 曲线
            Column(Modifier.weight(1.2f)) {
                MetricRow(rounds.lastOrNull(), nodes.size)
                Spacer(Modifier.height(12.dp))
                Panel("LOSS / ACC", Modifier.weight(1f)) {
                    MetricChart(rounds)
                }
            }
            Spacer(Modifier.width(12.dp))
            // 右：节点 + 日志
            Column(Modifier.weight(1f)) {
                Panel("CLUSTER NODES", Modifier.height(180.dp)) {
                    NodeList(events)
                }
                Spacer(Modifier.height(12.dp))
                Panel("EVENT LOG", Modifier.weight(1f)) {
                    EventLog(events)
                }
            }
        }
    }
}

@Composable
fun MetricRow(last: ClusterEvent?, nodeCount: Int) {
    Row {
        Metric("ROUND", last?.round?.toString() ?: "—", Modifier.weight(1f))
        Spacer(Modifier.width(8.dp))
        Metric("EVAL LOSS", last?.evalLoss?.let { "%.4f".format(it) } ?: "—",
               Modifier.weight(1f))
        Spacer(Modifier.width(8.dp))
        Metric("EVAL ACC", last?.evalAcc?.let { "%.2f%%".format(it * 100) } ?: "—",
               Modifier.weight(1f))
        Spacer(Modifier.width(8.dp))
        Metric("NODES", nodeCount.toString(), Modifier.weight(1f))
    }
}

@Composable
fun Metric(label: String, value: String, modifier: Modifier = Modifier) {
    Column(
        modifier
            .border(1.dp, MaheTheme.gridLine)
            .background(MaheTheme.panel)
            .padding(10.dp)
    ) {
        BasicText(label, style = MaheTheme.monoDim)
        Spacer(Modifier.height(4.dp))
        BasicText(value, style = MaheTheme.bigNumber)
    }
}

@Composable
fun Panel(title: String, modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    Column(modifier.border(1.dp, MaheTheme.gridLine).background(MaheTheme.panel)) {
        BasicText(
            "▸ $title",
            style = MaheTheme.mono.copy(color = MaheTheme.accent),
            modifier = Modifier.padding(8.dp),
        )
        Box(Modifier.fillMaxSize().padding(horizontal = 8.dp)) { content() }
    }
}

/** loss/acc 双折线（Canvas 手绘，细线网格风） */
@Composable
fun MetricChart(rounds: List<ClusterEvent>) {
    Canvas(Modifier.fillMaxSize().padding(bottom = 8.dp)) {
        if (rounds.size < 2) return@Canvas
        val losses = rounds.mapNotNull { it.evalLoss }
        val accs = rounds.mapNotNull { it.evalAcc }
        if (losses.size < 2) return@Canvas

        val w = size.width
        val h = size.height
        val maxLoss = (losses.maxOrNull() ?: 1.0).coerceAtLeast(1e-9)
        val n = losses.size

        // 网格
        for (i in 1..3) {
            val y = h * i / 4f
            drawLine(MaheTheme.gridLine, Offset(0f, y), Offset(w, y), 0.5f)
        }
        // loss（琥珀）
        var prev: Offset? = null
        losses.forEachIndexed { i, v ->
            val x = w * i / (n - 1)
            val y = h - (v / maxLoss * h * 0.9).toFloat() - h * 0.05f
            prev?.let { drawLine(MaheTheme.accent, it, Offset(x, y), 1.5f) }
            prev = Offset(x, y)
        }
        // acc（绿）
        prev = null
        accs.forEachIndexed { i, v ->
            val x = w * i / (accs.size - 1)
            val y = h - (v * h * 0.9).toFloat() - h * 0.05f
            prev?.let { drawLine(MaheTheme.ok, it, Offset(x, y), 1.5f) }
            prev = Offset(x, y)
        }
    }
}

@Composable
fun NodeList(events: List<ClusterEvent>) {
    val joined = events.filter { it.type == "node_join" }.mapNotNull { it.nodeName }
    val dead = events.filter { it.type == "node_dead" }.mapNotNull { it.nodeName }.toSet()
    LazyColumn {
        items(joined.distinct()) { name ->
            val alive = name !in dead
            BasicText(
                "${if (alive) "●" else "○"} $name  ${if (alive) "ALIVE" else "DEAD"}",
                style = MaheTheme.mono.copy(
                    color = if (alive) MaheTheme.textPrimary else MaheTheme.dead
                ),
                modifier = Modifier.padding(vertical = 2.dp),
            )
        }
    }
}

@Composable
fun EventLog(events: List<ClusterEvent>) {
    val state = rememberLazyListState()
    LaunchedEffect(events.size) { state.animateScrollToItem(maxOf(0, events.size - 1)) }
    LazyColumn(state = state) {
        items(events.takeLast(200)) { ev ->
            BasicText(ev.raw, style = MaheTheme.monoDim, maxLines = 1)
        }
    }
}
