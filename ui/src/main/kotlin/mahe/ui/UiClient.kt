package mahe.ui

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import java.io.BufferedReader
import java.io.BufferedWriter
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.Socket

/** 训练控制面事件（与 python mahe.ui_bridge 的 JSON Lines 对应） */
data class ClusterEvent(
    val type: String,
    val raw: String,
    val round: Int? = null,
    val evalLoss: Double? = null,
    val evalAcc: Double? = null,
    val nodeName: String? = null,
)

class UiClient(
    private val host: String = "127.0.0.1",
    private val port: Int = 47830,
) {
    private val _events = MutableStateFlow<List<ClusterEvent>>(emptyList())
    val events: StateFlow<List<ClusterEvent>> = _events

    private val _connected = MutableStateFlow(false)
    val connected: StateFlow<Boolean> = _connected

    private var socket: Socket? = null
    private var writer: BufferedWriter? = null

    fun connect(scope: CoroutineScope) {
        scope.launch(Dispatchers.IO) {
            while (true) {
                try {
                    val s = Socket(host, port)
                    socket = s
                    _connected.value = true
                    writer = BufferedWriter(OutputStreamWriter(s.getOutputStream()))
                    val reader = BufferedReader(InputStreamReader(s.getInputStream()))
                    reader.forEachLine { line -> handleLine(line) }
                } catch (_: Exception) {
                    _connected.value = false
                }
                // 断线重连
                try { Thread.sleep(2000) } catch (_: InterruptedException) { return@launch }
            }
        }
    }

    private fun handleLine(line: String) {
        // 轻量解析：不引入 JSON 库，按字段提取（协议字段稳定）
        fun field(name: String): String? {
            val key = "\"$name\":"
            val i = line.indexOf(key)
            if (i < 0) return null
            var j = i + key.length
            if (line[j] == '"') {
                val k = line.indexOf('"', j + 1)
                return line.substring(j + 1, k)
            }
            var k = j
            while (k < line.length && (line[k].isDigit() || line[k] in ".-+eE")) k++
            return line.substring(j, k)
        }
        val ev = ClusterEvent(
            type = field("type") ?: "unknown",
            raw = line,
            round = field("round")?.toIntOrNull(),
            evalLoss = field("eval_loss")?.toDoubleOrNull(),
            evalAcc = field("eval_acc")?.toDoubleOrNull(),
            nodeName = field("name"),
        )
        _events.value = (_events.value + ev).takeLast(500)
    }

    fun send(cmd: String) {
        try {
            writer?.apply { write(cmd); newLine(); flush() }
        } catch (_: Exception) {
            _connected.value = false
        }
    }
}
