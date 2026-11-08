package mahe.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.text.BasicText
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.delay

/** 550W 式启动自检序列：逐行系统日志 + 扫描线，结束后进入主界面。 */
@Composable
fun BootSequence(onDone: () -> Unit) {
    val lines = remember {
        listOf(
            "MAHE CORE .................. ONLINE",
            "KERNEL MODULE .............. AVX2 DISPATCH OK",
            "TRANSPORT LAYER ............ TCP FRAME PROTO v1",
            "DISCOVERY BEACON ........... UDP 47821 LISTENING",
            "SCHEDULER .................. WEIGHTED SHARD READY",
            "PARAMETER SERVER ........... STANDBY",
            "SECURITY HANDSHAKE ......... CLUSTER TOKEN LOADED",
            "WATCHDOG ................... 10000ms TIMEOUT ARMED",
            "",
            "ALL SYSTEMS NOMINAL.",
        )
    }
    var shown by remember { mutableStateOf(0) }
    var tick by remember { mutableStateOf(0) }

    LaunchedEffect(Unit) {
        while (shown < lines.size) {
            delay(if (lines[shown].isEmpty()) 400 else 140)
            shown++
        }
        delay(600)
        onDone()
    }
    LaunchedEffect(Unit) {
        while (true) { delay(50); tick++ }
    }

    Column(
        Modifier
            .fillMaxSize()
            .drawBehind {
                // 扫描线：一条自上而下的亮线
                val y = (tick * 8f) % (size.height + 200f) - 100f
                drawLine(
                    MaheTheme.accent.copy(alpha = 0.15f),
                    Offset(0f, y), Offset(size.width, y),
                    strokeWidth = 2f,
                )
                // 背景网格
                val step = 48f
                var gx = 0f
                while (gx < size.width) {
                    drawLine(MaheTheme.gridLine, Offset(gx, 0f), Offset(gx, size.height), 0.5f)
                    gx += step
                }
                var gy = 0f
                while (gy < size.height) {
                    drawLine(MaheTheme.gridLine, Offset(0f, gy), Offset(size.width, gy), 0.5f)
                    gy += step
                }
            }
            .padding(48.dp)
    ) {
        BasicText("MAHE // DISTRIBUTED TRAINING SYSTEM", style = MaheTheme.title)
        Spacer(Modifier.height(8.dp))
        BasicText("─".repeat(46), style = MaheTheme.monoDim)
        Spacer(Modifier.height(16.dp))
        lines.take(shown).forEach { line ->
            BasicText(
                if (line.endsWith("OK") || line.endsWith("NOMINAL.")) "▸ $line" else "  $line",
                style = if (line.endsWith("NOMINAL.")) MaheTheme.title else MaheTheme.mono,
            )
            Spacer(Modifier.height(4.dp))
        }
        if (shown < lines.size) {
            BasicText(if (tick % 2 == 0) "█" else " ", style = MaheTheme.mono)
        }
    }
}
